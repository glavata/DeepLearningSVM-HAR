from enum import Enum
import pandas as pd
import numpy as np
import cv2 as cv2
from sklearn.cluster import AffinityPropagation, DBSCAN
import os
import re
import sys
import scipy.ndimage.interpolation as inter

class Dataset(Enum):
    PKU_MMD = 0
    NTU_RGB = 1

class DataLoader:

    def __init__(self, dataset_type, directory, settings):
        if(dataset_type not in [item for item in Dataset]):
            raise Exception("Nonexistant dataset!")

        self.__dataset_type = Dataset(dataset_type)
        self.__dataset_name = str.lower(dataset_type.name)
        self.__dir = directory
        self.__settings = settings
        self.__opt_duration = None
        self.__feature_length = None


    def __extract_frame_distances(self, frames):
        frame_count = frames.shape[0]
        joint_coords_total = frames.shape[1]
        joint_c = int(joint_coords_total / 3)

        avg_joints = np.empty((frame_count, 3))
        avg_joints[:, 0] = np.mean(frames[:,0::3], 1)
        avg_joints[:, 1] = np.mean(frames[:,1::3], 1)
        avg_joints[:, 2] = np.mean(frames[:,2::3], 1)
    
        frames_res = frames.reshape(frame_count, joint_c, 3)
        joint_distances_final = np.linalg.norm(frames_res - avg_joints[:,np.newaxis,:], axis=2).reshape(frame_count, joint_c)

        return joint_distances_final

    def __scale_dataframe(self, frame, length_new):
        frame_feat = frame.shape[1]
        frame_length_old = frame.shape[0]
        #frame_new = cv2.resize(frame,(frame_feat, length), None, 0, 0, cv2.INTER_CUBIC)
        frame_new = inter.zoom(frame,[length_new/frame_length_old, 1])[:length_new]

        return frame_new

    def __normalize_dataframe(self, frame, torso_ind, neck_ind, j_count):

        torso_coords = frame[:,torso_ind:torso_ind+3]
        neck_coords = frame[:,neck_ind:neck_ind+3]
        norms = np.linalg.norm(neck_coords - torso_coords, axis=1)
        norms[norms==0.0] = 0.001
        torso_coords_mask = np.tile(torso_coords, (1,j_count))
        frame_normed = (frame - torso_coords_mask) / norms[:,None]

        return frame_normed

    def __form_distance_matrix(self, frame):
        ln = frame.shape[0]
        mat = np.zeros((ln,ln))

        for i in range(ln):
            for j in range(ln):
                if(i!=j):
                    mat[i,j] = np.linalg.norm(frame[i] - frame[j])

        return mat

    def __form_distance_vector(self, frame,ind):
        mat = self.__form_distance_matrix(frame)[ind]
        return mat
            

    def __frame_to_rgb(self, frame, joint_c):
        x_comp, y_comp, z_comp = frame[:,0::3], frame[:,1::3], frame[:,2::3]
        return np.dstack((x_comp,y_comp,z_comp))

    def load_data(self, reload=False):

        X = None
        y = np.array([], dtype="uint8")
        v = np.array([]) #activity duration

        settings_str = ''

        if(self.__settings != None):
            settings_str  = '_'+ '_'.join([str(obj[0]) + "-" + str(obj[1]) for obj in self.__settings.items()])
        ready_file_name =  "database_rdy\\" + self.__dataset_name + settings_str + ".npy"

        if(os.path.isfile(ready_file_name) and reload==False):
            with open(ready_file_name, 'rb') as f:
                rdy_arr = np.load(f, allow_pickle=True)
                X,y,z,v, = rdy_arr[0], rdy_arr[1], rdy_arr[2], rdy_arr[3]
                return X,y,z,v

        if(self.__dataset_type == Dataset.PKU_MMD):
            mult = np.array([12,16,18,21,24,26,27]) #multi person activities
           
            #max duration with multi acc - 759
            #269, 20, 1, 14
            flag = 0

            #103.0 median, 100 with multipers
            #117.52980231629392 avg, 114 
            X_mat = None
            X_vec = None
            mat_ind = np.tril_indices(100)


            for root, subFolders, files in os.walk(self.__dir + '//PKU_Skeleton_Renew', topdown=True):
                for file in files:
                    file_frames = self.__dir + '//PKU_Skeleton_Renew//' + file
                    file_labels = self.__dir + '//Train_Label_PKU_final//' + file

                    table_frames = pd.read_csv(file_frames, delimiter=' ', index_col=False, header=None)
                    table_labels = pd.read_csv(file_labels, delimiter=',', index_col=False, header=None)

                    for index, row in table_labels.iterrows():

                        class_label, frame_start, frame_end, confidence = int(row[0]), int(row[1]), int(row[2]), int(row[3])

                        if(frame_end <= frame_start or confidence != 2):
                            continue

                        class_frames_first = table_frames.loc[frame_start : frame_end].iloc[:,0:75]
                        class_frames_first = class_frames_first.to_numpy()
                        class_frames = class_frames_first

                        if(class_label in mult):
                            class_frames_sec = table_frames.loc[frame_start : frame_end].iloc[:,75:150]
                            class_frames_sec = class_frames_sec.to_numpy()
                            class_frames = np.vstack((class_frames,class_frames_sec))


                        scaled_frames = self.__scale_dataframe(class_frames,100)

                        normalized_frames = self.__normalize_dataframe(scaled_frames,16,2,25)
                        rgb_action = self.__frame_to_rgb(normalized_frames,25)
                        rgb_act_res = np.reshape(rgb_action,(1, rgb_action.shape[0],rgb_action.shape[1],rgb_action.shape[2]))
                        feat_vectors = self.__extract_frame_distances(scaled_frames)
                        dist_vec = self.__form_distance_vector(feat_vectors,mat_ind)

                        if flag == 0:
                            flag = 1
                            X_mat = rgb_act_res
                            X_vec = dist_vec
                        else:
                            X_mat = np.concatenate((X_mat, rgb_act_res))
                            X_vec = np.vstack((X_vec, dist_vec))

                        y = np.append(y, class_label)
            
            X = [X_mat, X_vec]

        v = np.unique(y)
        z = len(v)

        indices = list(range(0,z))
        dict_tmp = dict(zip(v.tolist(),indices))
        y = np.array([dict_tmp.get(i, -1) for i in y])

        with open(ready_file_name, 'wb') as f:
            np.save(f, np.array([X, y, z, v]))

        return X, y, z, v





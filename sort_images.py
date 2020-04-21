#!/usr/bin/python3.6

## import packages
import os
import sys
import cv2
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import numpy as np
import shutil
from imutils import paths
import json

DATASET = "./data/data_laundry/ValSet"
OUTPUT = "./data/data_laundry/ValSet"

#DATASET = "./data/iwildcam-2020-fgvc7/train"
#DATASET = "./data/animal_crops_224x224"

#DATASET = "./data/iwildcam-2020-fgvc7/test_by_seq_id"
#OUTPUT = "./data/iwildcam-2020-fgvc7/test_by_seq_id"

#DATASET = "./data/animal_crops_64x64/animal_crops_test_64x64"
#OUTPUT = "./data/animal_crops_64x64/animal_crops_test_64x64"

#DATASET = "./data/animal_crops_224x224/animal_crops_test_224x224"
#OUTPUT = "./data/animal_crops_224x224/animal_crops_test_224x224"

#DATASET = "./data/animal_crops_224x224/animal_crops_val_224x224"
#OUTPUT = "./data/animal_crops_224x224/animal_crops_val_224x224"

assert DATASET and OUTPUT

MODE = "train"
#TASK = "stats annotation dims & ARs"
#TASK = "sort"
#TASK = "count small classes"
#TASK = "resize"
TASK = "sort val video sequence"
#TASK = "stats on valPaths"

#MODE = "test"
#TASK = "sort test video sequence"
#TASK = "stats on testPaths"


# load in megadetector result
detector_results_images = pd.read_csv("./data/iwildcam2020_megadetector_results_images.csv")
print("[INFO] head of detector results...\n", detector_results_images.head())
print("cols = ", detector_results_images.columns)

# load in train_annotations &
if MODE == "train":
    trainAnno_annotations = pd.read_csv("./data/iwildcam2020_train_annotations_annotations.csv")
    trainAnno_categories = pd.read_csv("./data/iwildcam2020_train_annotations_categories.csv")
    trainAnno_images = pd.read_csv("./data/iwildcam2020_train_annotations_images.csv")

    print("[INFO] head of train annotations...\n", trainAnno_annotations.head())
    print("cols = ", trainAnno_annotations.columns)
    print("[INFO] head of train images...\n", trainAnno_images.head())
    print("cols = ", trainAnno_images.columns)
    print("[INFO] head of train categories...\n", trainAnno_categories.head())
    print("cols = ", trainAnno_categories.columns)
    print()

    if TASK == "stats annotation dims & ARs":
        print("[INFO] TAKS =", TASK)

        # rename detector_results col names
        detector_results_images = detector_results_images.rename(columns={"id" : "image_id"})

        # merge dfs by "image_id"
        mergedDF = pd.merge(detector_results_images, trainAnno_annotations,
                how="inner", on=["image_id"])
        
        # add an "image_id" column into trainAnno_images & drop ".jpg"
        trainAnno_images["image_id"] = trainAnno_images["file_name"]
        trainAnno_images["image_id"] = trainAnno_images["image_id"].apply(lambda x : x.split(".")[0])

        # merge on "image_id" again
        mergedDF = pd.merge(mergedDF, trainAnno_images, how="inner", on=["image_id"])

        # rename "id" in trainAnno_categories
        trainAnno_categories = trainAnno_categories.rename(columns={"id" : "category_id"})
        
        # merge on category id again
        mergedDF = pd.merge(mergedDF, trainAnno_categories, how="inner", on="category_id")

        # drop useless columns
        useful_cols = ['detections', 'image_id', 'max_detection_conf', 'datetime', \
       'width', 'height', 'file_name', 'category_id', 'name']
        finalDF = mergedDF[useful_cols] 
        finalDF.to_csv(os.path.sep.join([OUTPUT, "merged_train_annotations.csv"]), index=False)

        print("[INFO] peek the merged DF=\n", finalDF.head())

        ## create counter & loop over rows
        smallX_counter, smallY_counter, smallXY_counter = {}, {}, {}
        lateralAR_counter, verticalAR_counter = {}, {}
        time_counter = {}

        for index, row in tqdm(finalDF.iterrows()):
            detections = eval(row["detections"])
            time = row["datetime"].split(" ")[-1].split(":")[0] # str
            W = row["width"]
            H = row["height"]
            catname = row["name"]
        
            # loop over bboxes
            if not detections or W == 0 or H == 0:
                continue
            for idx, detection in enumerate(detections):
                x_rel, y_rel, w_rel, h_rel = detection['bbox']     
                anno_w, anno_h = w_rel * W, h_rel * H
                
                if anno_w == 0 or anno_h == 0:
                    continue

                # check dim  
                if anno_w < 128 and anno_h > 128:
                    smallX_counter[catname] = smallX_counter.get(catname, 0) + 1
                elif anno_h < 128 and anno_w > 128:
                    smallY_counter[catname] = smallY_counter.get(catname, 0) + 1
                elif anno_h < 128 and anno_w < 128:
                    smallXY_counter[catname] = smallXY_counter.get(catname, 0) + 1
                # check AR
                if anno_w / anno_h > 3:
                    lateralAR_counter[catname] = lateralAR_counter.get(catname, 0) + 1
                elif anno_h / anno_w > 3:
                    verticalAR_counter[catname] = verticalAR_counter.get(catname, 0) + 1
                pass
            # count time
            time_counter[time] = time_counter.get(time, 0) + 1

        # write out
        jsonpath1 = os.path.sep.join([OUTPUT, "x<128_stats.json"])
        jsonpath2 = os.path.sep.join([OUTPUT, "y<128_stats.json"])
        jsonpath3 = os.path.sep.join([OUTPUT, "xy<128_stats.json"])
        jsonpath4 = os.path.sep.join([OUTPUT, "128_xAR>3_stats.json"])
        jsonpath5 = os.path.sep.join([OUTPUT, "128_yAR>3_stats.json"])
        jsonpath6 = os.path.sep.join([OUTPUT, "time_stats.json"])
        counters = [smallX_counter, smallY_counter, smallXY_counter,
                lateralAR_counter, verticalAR_counter, time_counter]
        jsonpaths = [jsonpath1, jsonpath2, jsonpath3, jsonpath4, jsonpath5, jsonpath6]

        print("[INFO] writing jsons..")
        for jpath, counter in zip(jsonpaths, counters):
            with open(jpath, "w") as jfile:
                jfile.write(json.dumps(counter, separators=(",", ":")))
            jfile.close()

    if TASK == "sort":
        # sort values by train categories
        cat_df = trainAnno_categories[trainAnno_categories["count"] == 25]
        print("count == 20 categories are\n", cat_df.head())

        sys.exit()
        
        ## pick this guy
        #target = "philander opossum"
        #target_id = 67
        target = "hemigalus derbyanus"
        target_id = 142
        target_annotations = trainAnno_annotations[trainAnno_annotations["category_id"]==target_id]
        target_images = target_annotations["image_id"].values

        # try sort
        sorted(target_images)
        
        # copy all target images to a folder
        for img_id in tqdm(target_images):
            # create a folder
            folder = os.path.sep.join([OUTPUT, target])
            if not os.path.exists(folder):
                os.makedirs(folder)

            imgsrc = os.path.sep.join([DATASET, img_id + ".jpg"])
            image = cv2.imread(imgsrc)
            cv2.imshow("img", image)
            cv2.waitKey(0)

            imgdst = os.path.sep.join([folder, img_id + ".jpg"])
            shutil.copy(imgsrc, imgdst)
        pass

    if TASK == "count small classes":
        dataset = "./data/animal_crops_224x224"
        image_counter = {}

        for folder in os.listdir(dataset):
            image_num = len(list(paths.list_images(os.path.sep.join([dataset, folder]))))
            image_counter[folder] = image_num
            pass
        # sort by keys
        image_counter = {k : v for k, v in sorted(image_counter.items(), key=lambda x : x[1])}
        my_counter = {}     

        # distribute to Kai & me
        i = 0
        for name, value in image_counter.items():
            #if value <= 20:
            if 20 <= value <= 50:
                if i % 2 == 1:
                    #print(name, "=", value)
                    my_counter[name] = value
                i += 1
        print("[INFO] my work has classes = ", len(my_counter))

        # my workload
        outputfolder = "./data/data_laundry"
        classes = my_counter.keys()
        for cls in classes:
            # retrieve category id
            catid = trainAnno_categories[trainAnno_categories["name"] == cls].id.values[0]
            # get image names
            image_ids = trainAnno_annotations[trainAnno_annotations["category_id"] ==
                    catid].image_id.values
            # mkdirs 
            folder = os.path.sep.join([outputfolder, cls])
            if not os.path.exists(folder):
                os.makedirs(folder)
            # shift images to it
            for img_id in image_ids:
                imgsrc = os.path.sep.join(["./data/iwildcam-2020-fgvc7/train", img_id + ".jpg"])
                imgdst = os.path.sep.join([folder, img_id + ".jpg"])
                shutil.copy(imgsrc, imgdst)
        pass

    if TASK == "resize":
        print("[INFO] resizing crops...")
        dataset = "./data/animal_crops_224x224/data-add_Kai"
        imagePaths = list(paths.list_images(dataset))
        for path in tqdm(imagePaths):
            image = cv2.imread(path)
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(path, image)
            
        for path in tqdm(imagePaths):
            # hflip
            image = cv2.imread(path)
            image = cv2.flip(image, 1)

            path_seps = path.split(os.path.sep)
            imgname = path_seps[-1].split(".")[0]
            imgname += "_hf.jpg"
            path_seps[-1] = imgname
            newpath = "/".join(path_seps)
            cv2.imwrite(newpath, image)
            pass

    if TASK == "sort val video sequence":
        print("[INFO] TAKS = ", TASK)
        print("[INFO] sorting on", DATASET)

        """
        sort out by sequence
        """
        valAnno_images = pd.read_csv(os.path.sep.join([DATASET, "valAnno_images.csv"]))
        print("[INFO] there are %d unique seq_id" %  len(valAnno_images["seq_id"].unique()))

        unique_seq_id = valAnno_images["seq_id"].unique()
        seq_id_counter = {}
        for seqid in unique_seq_id:
            specific_seq_id = valAnno_images[valAnno_images["seq_id"] == seqid]
            seq_id_counter[seqid] = specific_seq_id.shape[0]

        for seq_id in tqdm(seq_id_counter.keys(), desc="sorting by seq_id"):
            df_seq_id = valAnno_images[valAnno_images["seq_id"] == seq_id]

            seq_id_folder = os.path.sep.join([OUTPUT, seq_id])
            if not os.path.exists(seq_id_folder):
               os.makedirs(seq_id_folder)

            for fname in df_seq_id["file_name"].values:
                imgsrc = os.path.sep.join([DATASET, fname])
                imgdst = os.path.sep.join([seq_id_folder, fname])
                #try:
                #    shutil.move(imgsrc, imgdst)
                #except:
                #    pass

            """
            sort out locations
            """
            locations = df_seq_id["location"].unique()
            for loc in locations:
                df_location = df_seq_id[df_seq_id["location"] == loc]
                loc_folder = os.path.sep.join([OUTPUT, seq_id, str(loc)])
    
                if not os.path.exists(loc_folder):
                    os.makedirs(loc_folder)
    
                for fname in df_location["file_name"].values:
                    imgsrc = os.path.sep.join([DATASET, seq_id, fname])
                    imgdst = os.path.sep.join([loc_folder, fname])
                    #try:
                    #    shutil.move(imgsrc, imgdst)
                    #except:
                    #    pass


            """
            sort out each location into clips via timestamp and image dimension
            """
            for loc in locations:
                df_location = df_seq_id[df_seq_id["location"] == loc]

                # cluster by timestamp
                df_location["datetime"] = pd.to_datetime(df_location["datetime"])
                df_datetime = df_location.drop(columns=["seq_num_frames", "id", "location", "frame_num", "seq_id"])
                # sort by timestamp
                df_datetime = df_datetime.sort_values(by=["datetime"], ascending=True, ignore_index=True)

                # get unique dimensions
                w_uniques = df_datetime["width"].unique()
                clip_index = 0

                # loop over each dim
                for w in w_uniques:
                    df_w = df_datetime[df_datetime["width"] == w].reset_index()
                    
                    if seq_id == "995a6da2-21bc-11ea-a13a-137349068a90":
                        clip_size = 10
                    else:
                        clip_size = 3

                    for i in range(0, df_w.shape[0], clip_size):
                        clip_folder = os.path.sep.join([OUTPUT, seq_id, str(loc), str(clip_index)])

                        if not os.path.exists(clip_folder):
                            os.makedirs(clip_folder)

                        for fname in df_w["file_name"].values[i : i + clip_size]:
                            imgsrc = os.path.sep.join([DATASET, seq_id, str(loc), fname])
                            imgdst = os.path.sep.join([clip_folder, fname])
                            #try:
                            #    shutil.move(imgsrc, imgdst)
                            #except:
                            #    pass

                        # update clip index
                        clip_index += 1
                    pass
                pass
            pass
        pass

    if TASK == "stats on valPaths":
        print("[INFO] stats on", DATASET)
        valPaths = list(paths.list_images(DATASET))
        
        pathsArray = [tpath.split(os.path.sep)[-4:] for tpath in valPaths]
        pathsArray = np.array(pathsArray)
        print(pathsArray.shape) 
        print(pathsArray[0])
        
        
        sorted_test_df = pd.DataFrame({"seq_id" : pathsArray[:, 0], "location" : pathsArray[:, 1], \
                "clip_index" : pathsArray[:, 2], "file_name" : pathsArray[:, 3]})
        sorted_test_df.to_csv(os.path.sep.join([OUTPUT, "sorted_val_images_directory.csv"]), index=False)
        pass




if MODE == "test":
    testAnno_categories = pd.read_csv("./data/iwildcam2020_train_annotations_categories_test.csv")
    testAnno_images = pd.read_csv("./data/iwildcam2020_train_annotations_images_test.csv")

    print("[INFO] head of test images...\n", testAnno_images.head())
    print("cols = ", testAnno_images.columns)
    print("shape = ", testAnno_images.shape)
    print("[INFO] head of test categories...\n", testAnno_categories.head())
    print("cols = ", testAnno_categories.columns)
    print()

    if TASK == "sort test video sequence":
        print("[INFO] TAKS = ", TASK)
        print("[INFO] sorting on", DATASET)
        print(testAnno_images.describe())


        """
        sort out by sequence
        """
        print("[INFO] there are %d unique seq_id" %  len(testAnno_images["seq_id"].unique()))

        unique_seq_id = testAnno_images["seq_id"].unique()        
        seq_id_counter = {}
        for seqid in unique_seq_id:
            specific_seq_id = testAnno_images[testAnno_images["seq_id"] == seqid]
            seq_id_counter[seqid] = specific_seq_id.shape[0]
        
        #seq_id_counter_df = pd.DataFrame({"seq_id" : list(seq_id_counter.keys()), "counts" :
        #    list(seq_id_counter.values())})
        #seq_id_counter_df = seq_id_counter_df.sort_values(by="counts", ascending=False)
        #seq_id_counter_df.to_csv("./data/data_laundry/unique_seq_id_counter.csv", index=False)
        #print(seq_id_counter_df.head())

        for seq_id in tqdm(seq_id_counter.keys(), desc="sorting by seq_id"):
            df_seq_id = testAnno_images[testAnno_images["seq_id"] == seq_id]

            seq_id_folder = os.path.sep.join([OUTPUT, seq_id])
            if not os.path.exists(seq_id_folder):
               os.makedirs(seq_id_folder)

            for fname in df_seq_id["file_name"].values:
                imgsrc = os.path.sep.join([DATASET, fname])
                imgdst = os.path.sep.join([seq_id_folder, fname])
                try:
                    shutil.move(imgsrc, imgdst)
                except:
                    pass


            """
            sort out locations
            """
            locations = df_seq_id["location"].unique()
            for loc in locations:
                df_location = df_seq_id[df_seq_id["location"] == loc]
                loc_folder = os.path.sep.join([OUTPUT, seq_id, str(loc)])
    
                if not os.path.exists(loc_folder):
                    os.makedirs(loc_folder)
    
                for fname in df_location["file_name"].values:
                    imgsrc = os.path.sep.join([DATASET, seq_id, fname])
                    imgdst = os.path.sep.join([loc_folder, fname])
                    try:
                        shutil.move(imgsrc, imgdst)
                    except:
                        pass

            """
            sort out each location into clips via timestamp and image dimension
            """
            for loc in locations:
                df_location = df_seq_id[df_seq_id["location"] == loc]

                # cluster by timestamp
                df_location["datetime"] = pd.to_datetime(df_location["datetime"])
                df_datetime = df_location.drop(columns=["seq_num_frames", "id", "location", "frame_num", "seq_id"])
                # sort by timestamp
                df_datetime = df_datetime.sort_values(by=["datetime"], ascending=True, ignore_index=True)

                # get unique dimensions
                w_uniques = df_datetime["width"].unique()
                clip_index = 0

                # loop over each dim
                for w in w_uniques:
                    df_w = df_datetime[df_datetime["width"] == w].reset_index()
                    
                    if seq_id == "995a6da2-21bc-11ea-a13a-137349068a90":
                        clip_size = 10
                    else:
                        clip_size = 3

                    for i in range(0, df_w.shape[0], clip_size):
                        clip_folder = os.path.sep.join([OUTPUT, seq_id, str(loc), str(clip_index)])

                        if not os.path.exists(clip_folder):
                            os.makedirs(clip_folder)

                        for fname in df_w["file_name"].values[i : i + clip_size]:
                            imgsrc = os.path.sep.join([DATASET, seq_id, str(loc), fname])
                            imgdst = os.path.sep.join([clip_folder, fname])
                            try:
                                shutil.move(imgsrc, imgdst)
                            except:
                                pass
                        # update clip index
                        clip_index += 1
                    pass
                pass
            pass
        pass

    if TASK == "stats on testPaths":
        print("[INFO] stats on", DATASET)
        testPaths = list(paths.list_images(DATASET))
        
        pathsArray = [tpath.split(os.path.sep)[-4:] for tpath in testPaths]
        pathsArray = np.array(pathsArray)
        print(pathsArray.shape) 
        print(pathsArray[0])
        
        
        sorted_test_df = pd.DataFrame({"seq_id" : pathsArray[:, 0], "location" : pathsArray[:, 1], \
                "clip_index" : pathsArray[:, 2], "file_name" : pathsArray[:, 3]})
        sorted_test_df.to_csv(os.path.sep.join([OUTPUT, "sorted_test_images_directory.csv"]), index=False)




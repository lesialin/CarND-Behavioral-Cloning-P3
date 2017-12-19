import csv
import cv2
import numpy as np
import sys
import os
import pandas
import glob
import sklearn

#Modify the data directory in the log file
def modify_logfile(log_path,train_path):
	lines = []
	with open(log_path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			for i in range(3):
				source_path = line[i]
				filename = os.path.split(source_path)[-1] 
				current_path = train_path +'/IMG/'+filename
				line[i] = current_path
			lines.append(line)
	#modified log filename
	log_filename = os.path.split(log_path)[-1] 
	last_slash_idx = log_path.rfind('/')
	log_dir = log_path[0:last_slash_idx]
	md_log_path = log_dir + '/md_' + log_filename
	#write modified log data into new logfile
	pd = pandas.DataFrame(lines)
	pd.to_csv(md_log_path,header=False,index=False)

def modify_all_logfile(dataset_dir):
	train_data_dirs = os.listdir(dataset_dir)
	for train_dir in train_data_dirs:
		log_path = dataset_dir + '/'+ train_dir + '/driving_log.csv'
		train_path = dataset_dir + '/'+ train_dir
		modify_logfile(log_path,train_path)



def augment_train_data(source_dir,mode='flip'):
	#remove slash in the end of the string
	while source_dir[-1]=='/':
		source_dir = source_dir[0:-1]
	#get the directories name
	data_dir = source_dir.split('/')[-1]	
	last_slash_idx = source_dir.rfind('/')
	root_path = source_dir[0:last_slash_idx]
	#create directory for augmented data 
	augmented_data_dir = root_path + '/' + data_dir +'_'+ mode
	augmented_data_image_dir = root_path + '/' + data_dir +'_'+ mode + '/IMG'
	if not os.path.exists(augmented_data_dir):
		os.mkdir(augmented_data_dir)
		os.mkdir(augmented_data_image_dir)

	#log_filename
	log_filename = 'md_driving_log.csv'
	#origin log filename 
	source_log_filename = source_dir + '/' + log_filename
	#augmented log filename
	augment_log_filename = augmented_data_dir + '/' + log_filename
	#store the augmented log data 
	lines = []
	if mode == 'flip':
		with open(source_log_filename) as csvfile:
			reader = csv.reader(csvfile)
			print('write the augmented image to')
			for line in reader:
				origin_image_path = line[0]
				for i in range(2,-1,-1):
					img_filename = line[i].split('/')[-1]
					img_path = augmented_data_dir + '/IMG/' + img_filename
					line[3] = -float(line[3])
					line[i] = img_path
				lines.append(line)	
				print(img_path)
				image = cv2.imread(origin_image_path)
				#augment here!!
				#flip image 180 degress, here I only flip center image
				image = cv2.flip(image,1)
				cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
				
	elif mode =='shift_right':
		pass

	elif mode == 'shift_left':
		pass
	#write modified log data into new logfile
	pd = pandas.DataFrame(lines)
	pd.to_csv(augment_log_filename,header=False,index=False)

def augment_all_train_data(dataset_dir):
	train_data_dirs = os.listdir(dataset_dir)
	for train_dir in train_data_dirs:
		train_path = dataset_dir + '/'+ train_dir
		augment_train_data(train_path)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread( batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def make_samples(dataset_dir):
	samples = []
	train_data_dirs = os.listdir(dataset_dir)
	for train_dir in train_data_dirs:
		log_path = dataset_dir + '/'+ train_dir + '/md_driving_log.csv'
		with open(log_path) as csvfile:
		    reader = csv.reader(csvfile)
		    for line in reader:
		        samples.append(line)

	return samples

		

#make dataset to npy files
def make_datset(log_filename,image_dir,feature_filename,label_filename):
	lines = []
	with open(log_filename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	images = []
	measurements = []
	for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = image_dir+filename
		image = cv2.imread(current_path)		
		#normalization
		images.append(image)
		steer_angel = float(line[3])
		measurements.append(steer_angel)
		#augementation
		image = cv2.flip(image,1)
		images.append(image)
		steer_angel = -float(line[3])
		measurements.append(steer_angel)

	X_train = np.array(images)
	y_train = np.array(measurements)
	np.save(feature_filename,X_train)
	np.save(label_filename,y_train)

def read_data(feature_filename,label_filename):
	X_train = np.load(feature_filename)
	y_train = np.load(label_filename)
	return X_train,y_train



if __name__ == '__main__':

	modify_all_logfile('data')
	augment_all_train_data('data')
	'''
	if len(sys.argv) !=5:
		print('Usage:python input.py log_filename image_dir feature_filename label_filename')
		exit()
	else:
		log_filename= sys.argv[1]
		image_dir = sys.argv[2]
		feature_filename = sys.argv[3]
		label_filename = sys.argv[4]
		make_datset(log_filename,image_dir,feature_filename,label_filename)
	'''
from FR2 import face_recognition
import argparse
threshold = 1
database = "database.npy"
faceCascade= "haarcascades/haarcascade_frontalface_default.xml"

def main(model = "haarcascades/haarcascade_frontalface_default.xml", source = "image", archive = None, threshold = 1, source_path="T01T1CLGP3Q-U02F9EJP5GC-73c746e07316-192.jpg", save_path = None):
	"""
	receive args listed in parse_opt
	return and print the archived name of any recognized face
	"""
	if archive:
		database_exist = True
	else:
		database_exist = False
	face_recog = face_recognition(threshold= threshold, haarcascades = model,
	database_exist = database_exist, database_path = archive)

	face_recog.face_detection(source = source,path=source_path, save_path = save_path)

def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', nargs='+', type=str, default="haarcascades/haarcascade_frontalface_default.xml", help='model path(s)')
	parser.add_argument('--source', type=str, default="image", help='image/video/webcam')	
	parser.add_argument('--archive', type=str, default='database.npy', help='(optional) existing archive path')
	parser.add_argument('--threshold', type=float, default=1, help='confidence threshold')
	parser.add_argument('--source_path', type=str, default="T01T1CLGP3Q-U02F9EJP5GC-73c746e07316-192.jpg", help='input source files for detection')
	parser.add_argument('--save_path', type=str, default=None, help='save path for result demo video')	
	opt = parser.parse_args()
	return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
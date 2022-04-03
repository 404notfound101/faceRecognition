from FR2 import face_recognition
import argparse
database = "database.npy"
threshold = 1
faceCascade= "haarcascades/haarcascade_frontalface_default.xml"

def main(model = "haarcascades/haarcascade_frontalface_default.xml", name = "prashant", archive = None, threshold = 1, img_count=5):
	if archive:
		database_exist = True
	else:
		database_exist = False
	face_recog = face_recognition(threshold= threshold, haarcascades = model,
		database_exist = database_exist, database_path = archive)

	face_recog.train(name = name, img_count = img_count)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default="haarcascades/haarcascade_frontalface_default.xml", help='model path(s)')
    parser.add_argument('--name', type=str, default="prashant", help='define a name for new recognition source')	
    parser.add_argument('--archive', type=str, default='database.npy', help='(optional) existing archive path')
    parser.add_argument('--threshold', type=float, default=1, help='confidence threshold')
    parser.add_argument('--img_count', type=int, default=5, help='number of frames catched from the webcam for archiving new recognition source')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
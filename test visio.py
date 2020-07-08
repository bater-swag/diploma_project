from keras.models import load_model
import argparse
import pickle
import cv2

print("[INFO] loading network and label binarizer...")
model = load_model('/home/bater/PycharmProjects/нейро/keras-tutorial/output/smallvggnet.model')
lb = pickle.loads(open('/home/bater/PycharmProjects/нейро/keras-tutorial/output/smallvggnet_lb.pickle', "rb").read())
cap = cv2.VideoCapture('4.mp4')
writer = None
flatten = -1
while(1):
	_, image = cap.read()
	output = image.copy()
	image = cv2.resize(image, (64, 64))
	image = image.astype("float") / 255.0

	if flatten > 0:
		image = image.flatten()
		image = image.reshape((1, image.shape[0]))

	# в противном случае мы работаем с CNN -- не сглаживаем изображение
	# и просто добавляем размер пакета
	else:
		image = image.reshape((1, image.shape[0], image.shape[1],
							   image.shape[2]))


	# распознаём изображение
	preds = model.predict(image)

	# находим индекс метки класса с наибольшей вероятностью
	# соответствия
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	# делаем предсказание на изображении
	preds = model.predict(image)
	print(preds)

	# находим индекс метки класса с наибольшей вероятностью
	# соответствия
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	# рисуем метку класса + вероятность на выходном изображении
	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
	cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
				(0, 0, 255), 2)
	(H, W) = output.shape[:2]
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('/home/bater/PycharmProjects/нейро/keras-tutorial/output/yyy2.avi', fourcc, 30,
								 (W, H), True)
	# write the output frame to disk
	writer.write(output)
	# показываем выходное изображение
	cv2.imshow("Image", output)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
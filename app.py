import cv2
import os
import imutils
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():

    if request.method == 'POST':
        post_image = request.files['image']

        path = os.path.join('static', post_image.filename)
        post_image.save(path)

        image = cv2.imread(path)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        image = imutils.resize(image,
                               width=min(500, image.shape[1]))

        (humans, _) = hog.detectMultiScale(image,
                                           winStride=(5, 5),
                                           padding=(3, 3),
                                           scale=1.21)

        for (x, y, w, h) in humans:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 2)

        cv2.imwrite(os.path.join(path), image)

        human_detected = str(len(humans))

        return render_template('index.html', path=path, hdetected=human_detected)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

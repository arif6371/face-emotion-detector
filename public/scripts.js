console.log(faceapi);

const run = async () => {
  // Loading the models
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });

  const videoFeedEl = document.getElementById("video-feed");
  videoFeedEl.srcObject = stream;

  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
    faceapi.nets.ageGenderNet.loadFromUri("./models"),
    faceapi.nets.faceExpressionNet.loadFromUri("./models"),
    faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
    faceapi.nets.faceLandmark68TinyNet.loadFromUri("./models"),
  ]);

  const canvas = document.getElementById("canvas");
  canvas.style.position = "absolute";
  canvas.style.left = `${videoFeedEl.offsetLeft}px`;
  canvas.style.top = `${videoFeedEl.offsetTop}px`;
  canvas.height = videoFeedEl.height;
  canvas.width = videoFeedEl.width;

  const context = canvas.getContext("2d");

  // Reference face data (for recognition example)
  const refFace = await faceapi.fetchImage(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Michael_Jordan_in_2014.jpg/220px-Michael_Jordan_in_2014.jpg"
  );
  let refFaceAiData = await faceapi
    .detectAllFaces(refFace)
    .withFaceLandmarks()
    .withFaceDescriptors();
  let faceMatcher = new faceapi.FaceMatcher(refFaceAiData);

  // Emotion tracking history
  let emotionHistory = [];

  // Real-time face detection and landmarking
  setInterval(async () => {
    const options = new faceapi.TinyFaceDetectorOptions({
      inputSize: 512,
      scoreThreshold: 0.5,
    });
    let faceAIData = await faceapi
      .detectAllFaces(videoFeedEl, options)
      .withFaceLandmarks(true)
      .withFaceDescriptors()
      .withAgeAndGender()
      .withFaceExpressions();

    faceAIData = faceapi.resizeResults(faceAIData, videoFeedEl);

    // Clear previous canvas drawings
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detections, landmarks, and expressions
    faceapi.draw.drawDetections(canvas, faceAIData);
    faceapi.draw.drawFaceLandmarks(canvas, faceAIData);
    faceapi.draw.drawFaceExpressions(canvas, faceAIData);

    faceAIData.forEach((face) => {
      const {
        age,
        gender,
        genderProbability,
        detection,
        descriptor,
        expressions,
        landmarks,
      } = face;

      // Draw age and gender info
      const genderText = `${gender} - ${Math.round(genderProbability * 100)}%`;
      const ageText = `${Math.round(age)} years`;
      const textField = new faceapi.draw.DrawTextField(
        [genderText, ageText],
        detection.box.topRight
      );
      textField.draw(canvas);

      // Recognize face
      let label = faceMatcher.findBestMatch(descriptor).toString();
      let drawOptions = { label: "Jordan" };
      if (label.includes("unknown")) {
        drawOptions = { label: "Unknown..." };
        // Optionally blur unknown faces
        context.globalAlpha = 0.5;
        context.drawImage(
          canvas,
          face.detection.box.x,
          face.detection.box.y,
          face.detection.box.width,
          face.detection.box.height,
          face.detection.box.x,
          face.detection.box.y,
          face.detection.box.width / 5,
          face.detection.box.height / 5
        );
        context.globalAlpha = 1.0;
      }

      // Draw the bounding box
      const drawBox = new faceapi.draw.DrawBox(detection.box, drawOptions);
      drawBox.draw(canvas);

      // Track and log emotions
      const topEmotion = Object.keys(expressions).reduce((a, b) =>
        expressions[a] > expressions[b] ? a : b
      );
      emotionHistory.push({ timestamp: Date.now(), emotion: topEmotion });
      console.log(emotionHistory);

      // Estimate head pose based on key facial landmarks (nose, eyes)
      const nose = landmarks.getNose();
      const leftEye = landmarks.getLeftEye();
      const rightEye = landmarks.getRightEye();

      // Example: Calculate the horizontal tilt angle
      const eyeDist = rightEye[0].x - leftEye[0].x;
      const noseTilt = (nose[0].x - (leftEye[0].x + eyeDist / 2)) / eyeDist;
      console.log("Head Pose (horizontal tilt):", noseTilt);

      // Trigger an event if recognized person is Jordan
      if (label.includes("Jordan")) {
        alert("Michael Jordan recognized!");
      }
    });
  }, 200);
};

run();

package com.kotlintask.touchless_fingerprint

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.*
import androidx.annotation.OptIn
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import org.json.JSONException
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScanPrints : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var objectDetector: ObjectDetector
    private lateinit var handMask: ImageView
    private lateinit var btn_back: Button
    private lateinit var progressMessage: TextView
    private lateinit var progress_spinner: ProgressBar
    private var leftHand: Boolean = false
    private var rightHand: Boolean = false
    private var leftThumb: Boolean = false
    private var rightThumb: Boolean = false
    private lateinit var guidance_text: TextView
    private lateinit var guidance_bars: LinearLayout
    private lateinit var guidance_bar_1: View
    private lateinit var guidance_bar_2: View
    private lateinit var guidance_bar_3: View
    private var isOptimalForCapture = false
    private var autoCaptureHandler: Handler? = null
    private var autoCaptureRunnable: Runnable? = null
    private lateinit var imageCapture: ImageCapture
    private var isCountingDown = false
    private var isProcessingCapture = false
    lateinit var detector: TFLiteObjectDetection

    private lateinit var countdownTimer: TextView
    var is_time_called=false
    lateinit var thumb_detector: TFLiteObjectDetection
    lateinit var test_image_view: ImageView
    var boxSize: Int = 0
    var baseURL = "https://3432-2407-d000-a-7997-10ee-b314-8d0e-90b5.ngrok-free.app/"
    private val full_urls = mutableListOf<String>()
    var handinfo = "No Hand"
    var bounding_boxes= mutableListOf<RectF>()
    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_scanprints)
//        test_image_view=findViewById(R.id.newImg)
        initializeUI()

        detector= TFLiteObjectDetection(
            "best_float32.tflite", "labels.txt", assets
        );

        thumb_detector=TFLiteObjectDetection(
            "best_float32_thumb_dec.tflite", "labels.txt", assets
        );
        // Initialize Object Detector
        val options = ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE) // Streaming mode for real-time detection
            .enableMultipleObjects()
            .enableClassification() // Enable object classification
            .build()
        objectDetector = ObjectDetection.getClient(options)

        // Initialize Camera Executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        startCamera(false)
    }

    private fun initializeUI() {


        previewView = findViewById(R.id.preview_view)
        btn_back = findViewById(R.id.btn_back)
        handMask = findViewById(R.id.hand_mask)
        progressMessage = findViewById(R.id.progress_message)
        progress_spinner = findViewById(R.id.progress_spinner)
        guidance_text = findViewById(R.id.guidance_text)
        guidance_bars = findViewById(R.id.guidance_bars)
        guidance_bar_1 = findViewById(R.id.guidance_bar_1)
        guidance_bar_2 = findViewById(R.id.guidance_bar_2)
        guidance_bar_3 = findViewById(R.id.guidance_bar_3)
        countdownTimer = findViewById(R.id.countdown_timer)

        handMask.setImageResource(R.drawable.left_hand_mask)
        handMask.visibility = View.VISIBLE
        leftHand = true
        rightHand = false
        leftThumb = false
        rightThumb = false
        // Back button logic
        btn_back.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        // Button clicks for masks
        findViewById<Button>(R.id.btn_left_hand).setOnClickListener {
            handMask.setImageResource(R.drawable.left_hand_mask)
            handMask.visibility = View.VISIBLE
            leftHand = true
            rightHand = false
            leftThumb = false
            rightThumb = false
            startCamera(false)
        }

        findViewById<Button>(R.id.btn_right_hand).setOnClickListener {
            handMask.setImageResource(R.drawable.right_hand_mask)
            handMask.visibility = View.VISIBLE
            leftHand = false
            rightHand = true
            leftThumb = false
            rightThumb = false
            startCamera(false)
        }

        findViewById<Button>(R.id.btn_left_thumb).setOnClickListener {
            handMask.setImageResource(R.drawable.left_thumb)
            handMask.visibility = View.VISIBLE
            leftHand = false
            rightHand = false
            leftThumb = true
            rightThumb = false
            startCamera(true)
        }

        findViewById<Button>(R.id.btn_right_thumb).setOnClickListener {
            handMask.setImageResource(R.drawable.right_thumb)
            handMask.visibility = View.VISIBLE
            leftHand = false
            rightHand = false
            leftThumb = false
            rightThumb = true
            startCamera(true)
        }
    }

    private fun startCamera(isthumb:Boolean) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            var rotation: Int
            try {
                // Get the rotation value from the display, if available
                rotation = previewView.display?.rotation ?: Surface.ROTATION_0  // Use Surface.ROTATION_0 if null
            } catch (e: Exception) {
                Log.e("Camera", "Error retrieving display rotation: ${e.message}")
                rotation = Surface.ROTATION_0  // Default to Surface.ROTATION_0 in case of error
            }

            // Ensure the rotation value is one of the valid Surface.ROTATION_* constants
            rotation = when (rotation) {
                Surface.ROTATION_0, Surface.ROTATION_90, Surface.ROTATION_180, Surface.ROTATION_270 -> rotation
                else -> Surface.ROTATION_0 // Fallback to Surface.ROTATION_0 if the value is not valid
            }


            // Preview use case
            val preview = Preview.Builder()
                .setTargetRotation(rotation)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }


            // ImageCapture use case
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY) // Ensure high-quality captures
                .setFlashMode(ImageCapture.FLASH_MODE_ON) // Default flash mode
                .setTargetRotation(rotation)
                .build()

            // ImageAnalysis use case
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // Prevent frame backlog
                .setTargetRotation(rotation)
                .build()
                .also { analysisUseCase ->
                    analysisUseCase.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy) // Analyze each frame
                    }
                }

            // Unbind all use cases and bind the new ones
            try {
                cameraProvider.unbindAll()
                val camera=cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )

                camera.cameraControl.enableTorch(isthumb)

            } catch (e: Exception) {
                Log.e("Camera", "Failed to bind use cases: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    fun imageToBitmap(image: ImageProxy): Bitmap {
        // Get the YUV_420_888 planes
        val planes = image.planes
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y data
        yBuffer[nv21, 0, ySize]

        // Interleave UV data
        val uvPixelStride = planes[1].pixelStride
        val uvRowStride = planes[1].rowStride

        val uBytes = ByteArray(uSize)
        val vBytes = ByteArray(vSize)
        uBuffer[uBytes]
        vBuffer[vBytes]

        for (row in 0 until planes[1].buffer.capacity() / uvRowStride) {
            var col = 0
            while (col < uvRowStride) {
                val uvIndex = row * uvRowStride + col
                val nvIndex =
                    ySize + 2 * (row * (uvRowStride / uvPixelStride) + (col / uvPixelStride))
                if (nvIndex + 1 < nv21.size && uvIndex < uBytes.size && uvIndex < vBytes.size) {
                    nv21[nvIndex] = vBytes[uvIndex] // V
                    nv21[nvIndex + 1] = uBytes[uvIndex] // U
                }
                col += uvPixelStride
            }
        }

        // Convert NV21 to Bitmap using YuvImage
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, outputStream)
        val jpegBytes = outputStream.toByteArray()

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    fun rotateBitmap(bitmap: Bitmap?, degrees: Int): Bitmap? {
        if (bitmap == null) {
            return null
        }

        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat()) // Rotate the bitmap by specified degrees
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }



    @OptIn(ExperimentalGetImage::class)
    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image

        var bitmap= imageToBitmap(imageProxy)
        if (leftThumb || rightThumb) {

            val result1=thumb_detector.runInference(rotateBitmap(bitmap,90),  true,handMask.drawable.intrinsicWidth,handMask.drawable.intrinsicHeight,leftHand)
            boxSize=result1.first
            bounding_boxes=result1.second
        }
        else
        {
            val result2=detector.runInference(rotateBitmap(bitmap,90),false,0,0,leftHand)
            boxSize=result2.first
            bounding_boxes=result2.second
        }
        val ellipseOverlayView = findViewById<EllipseOverlayView>(R.id.ellipse_overlay)
        ellipseOverlayView.boundingBoxes = bounding_boxes
        ellipseOverlayView.invalidate()

//
        if (mediaImage != null) {

            val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            objectDetector.process(inputImage)
                .addOnSuccessListener { detectedObjects ->

                    if (detectedObjects.isNotEmpty()) {

                        if ((leftHand || rightHand) && boxSize > 0) {
                            handleDetectedObjects(detectedObjects)
                        }
                        else {
                            if ((leftThumb || rightThumb) && boxSize>2) {
                                handleDetectedObjects(detectedObjects)
                            }
                            else
                            {
                                runOnUiThread{
                                    updateUI("No Hand", "Unknown Distance", 0)
                                }
                            }

                        }

                    } else {
//                        Log.d("ObjectDetection", "No objects detected.")
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("ObjectDetection", "Error detecting objects: ${e.message}")
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        }
    }

    private fun handleDetectedObjects(detectedObjects: List<DetectedObject>) {
        for (detectedObject in detectedObjects) {
            val boundingBox = detectedObject.boundingBox
            val trackingId = detectedObject.trackingId


            // Check for object classification (if enabled)
            for (label in detectedObject.labels) {
                val text = label.text
                val confidence = label.confidence
            }

            // Custom checks
            performCustomChecks(boundingBox, trackingId)
        }
    }

    private fun performCustomChecks(boundingBox: android.graphics.Rect, trackingId: Int?) {
        // 1. Check if a hand is present (bounding box exists)
        val handDetected = boundingBox.width() > 0 && boundingBox.height() > 0
        if (handDetected) {
            Log.d("HandCheck", "Hand detected with Tracking ID: $trackingId")
        } else {
            Log.d("HandCheck", "No hand detected.")
            return
        }

        // 2. Check for left or right hand (custom logic based on bounding box or labels)
        val isHand = when {
            leftHand -> 1
            rightHand -> 2
            leftThumb -> 3
            rightThumb -> 4
            else -> 5
        }

        val handType = if (isHand == 1)
                        {
                            handinfo = "Left Hand"
                        }
                        else if (isHand == 2)
                        {
                            handinfo = "Right Hand"
                        }
                        else if (isHand == 3)
                        {
                            handinfo = "Left Thumb"
                        }
                        else if (isHand == 4)
                        {
                            handinfo = "Right Thumb"
                        }
                        else
                        {
                            handinfo = "No Hand"
                            updateUI(handinfo, "Unknown Distance", 0)
                        }

        // 3. Determine distance based on bounding box size
        val handDistance = when {
            boundingBox.width() > 480 -> "Too Close"
            boundingBox.width() in 250..480 -> "Optimal"
            boundingBox.width() < 250 -> "Too Far"
            else -> "Unknown"
        }
        val handDistanceInPixels = boundingBox.width()

        // Display results on the UI
        updateUI(handinfo, handDistance, handDistanceInPixels)
    }

    private fun updateUI(handType: String, handDistance: String, handDistanceInPixels: Int) {



        if (boxSize==-2 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.red))
            val message = "No Hand Detected"
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.red))
            guidance_bar_2.visibility = View.INVISIBLE
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else if (boxSize == 0 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.yellow))
            val message = "Hand alignment not correct. Please adjust."
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else if (boxSize == -1 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.yellow))
            val message = "Hand alignment not correct. Please adjust."
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else if (boxSize == -10 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.red))
            val message = "No thumb Detected"
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.red))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.red))
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else if (boxSize == -100 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.yellow))
            val message = "Move thumb to the Correct Position"
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else if (boxSize == -50 && !is_time_called) {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.yellow))
            val message = "Hand alignment not correct. Please adjust."
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.yellow))
            guidance_bar_3.visibility = View.INVISIBLE
            return
        }
        else
        {
            guidance_text.setTextColor(ContextCompat.getColor(this, R.color.green))
            val message = "Optimal Distance, Stay Still"
            guidance_text.text = message
            guidance_text.visibility = TextView.VISIBLE
            guidance_bars.visibility = LinearLayout.VISIBLE
            guidance_bar_1.visibility = View.VISIBLE
            guidance_bar_1.setBackgroundColor(ContextCompat.getColor(this, R.color.green))
            guidance_bar_2.visibility = View.VISIBLE
            guidance_bar_2.setBackgroundColor(ContextCompat.getColor(this, R.color.green))
            guidance_bar_3.visibility = View.VISIBLE
            guidance_bar_3.setBackgroundColor(ContextCompat.getColor(this, R.color.green))
            if (!isOptimalForCapture) {
                isOptimalForCapture = true

                startCountdown()
            }
        }
    }

    private fun startCountdown() {
        if(!is_time_called)
        {
            is_time_called=true
            countdownTimer.visibility = View.VISIBLE // Make the countdown timer visible

            // Countdown logic
            val countdownHandler = Handler(Looper.getMainLooper())
            var secondsLeft = 4
            countdownTimer.text = secondsLeft.toString() // Set initial value

            val countdownRunnable = object : Runnable {
                override fun run() {
                    secondsLeft--
                    if (secondsLeft > 0) {
                        countdownTimer.text = secondsLeft.toString()
                        countdownHandler.postDelayed(this, 1000) // Run again after 1 second


                    } else {
                        countdownTimer.visibility = View.GONE // Hide the countdown timer
                        if (boxSize>0)
                        {
                            startAutoCaptureCountdown()
                        }
                        else
                        {
                            isOptimalForCapture = false
                            isCountingDown = false
                            is_time_called=false
                        }
                    }
                }
            }
            countdownHandler.post(countdownRunnable) // Start the countdown
        }
        else{
            guidance_text.text="Optimal Distance, Stay Still"
        }

    }

    private fun startAutoCaptureCountdown() {
        if (isCountingDown || isProcessingCapture) {
            return
        }
        isCountingDown = true
        autoCaptureHandler = Handler(Looper.getMainLooper())
        autoCaptureRunnable = Runnable {
            if (isOptimalForCapture && !isProcessingCapture) {
                isProcessingCapture = true
                resetAutoCapture()
                autoCaptureImage() // Call your auto-capture image function here
            }
            isCountingDown = false
        }
        // Delay for 2 seconds
        autoCaptureHandler?.postDelayed(autoCaptureRunnable!!, 2000)
    }

    private fun resetAutoCapture() {
        isOptimalForCapture = false
        isCountingDown = false
        autoCaptureHandler?.removeCallbacks(autoCaptureRunnable!!)
    }

    private fun autoCaptureImage() {
        // Ensure ImageCapture is initialized
        if (!::imageCapture.isInitialized) {
            resetAutoCapture()
            isProcessingCapture = false
            return
        }

        // Display spinner and processing message
        showProcessingState("Processing...")
        // File to save the image
        val jpegFile = File(externalMediaDirs.firstOrNull(), "captured_image.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(jpegFile).build()

        // Enable flash
        imageCapture.flashMode = ImageCapture.FLASH_MODE_ON

        // Take the picture
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {

                    removeExifRotation(jpegFile)
                    sendImageToServer(jpegFile)

                    // Stop the camera
                    stopCamera()

                    // Update UI to indicate processing complete
                    showProcessingState("Extracting Fingerprints")
                    isProcessingCapture = false
                    is_time_called=false
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("AutoCapture", "Failed to capture image: ${exception.message}")
                    showProcessingState("Error capturing image.")
                    isProcessingCapture = false
                    is_time_called=false
                }
            }
        )
    }

    private fun removeExifRotation(imageFile: File) {
        try {
            val exif = androidx.exifinterface.media.ExifInterface(imageFile.absolutePath)
            exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL.toString())
            exif.saveAttributes()
        } catch (e: IOException) {

        }
    }


    private fun showImagesInRecyclerView() {
        val imageRecyclerView = findViewById<RecyclerView>(R.id.image_recycler_view)
        imageRecyclerView.layoutManager = LinearLayoutManager(this)
        val imageUrls = mutableListOf<String>()

        // Fetch the images from the NGROK server
            runOnUiThread {
                imageUrls.clear()
                imageUrls.addAll(full_urls)

                // Set up the RecyclerView
                val adapter = ImageAdapter(imageUrls,handinfo) { imageUrl ->
                    // On image click, launch full-screen activity
                    val intent = Intent(this, FullScreenImageActivity::class.java)
                    intent.putExtra("image_url", imageUrl)
                    startActivity(intent)
                }
                imageRecyclerView.adapter = adapter
                imageRecyclerView.visibility = View.VISIBLE
            }
    }

    // Function to parse JSON response and extract image URLs
    private fun parseJsonImageUrls(jsonObject: JSONObject): List<String> {
        if (jsonObject == null) return emptyList()
        val urls = mutableListOf<String>()
        try {

            // Check for the "paths" key
            if (jsonObject.has("paths")) {
                val pathsArray = jsonObject.getJSONArray("paths")

                // Iterate through the array and extract URLs
                for (i in 0 until pathsArray.length()) {
                    val url = pathsArray.getString(i)
                    val fullUrl = "$baseURL${url}"
                    urls.add(fullUrl)
                }
            } else {
                Log.e("AutoCapture", "JSON does not contain 'paths' key.")
            }
        } catch (e: JSONException) {
            Log.e("AutoCapture", "Failed to parse JSON: ${e.message}")
            runOnUiThread {
                Toast.makeText(applicationContext, "JSON Parse Failure: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
        full_urls.clear()
        full_urls.addAll(urls)
        return urls
    }

    private fun sendImageToServer(imageFile: File) {
        val ngrokUrl = baseURL
        val activeHand = when {
            leftHand -> "left"
            rightHand -> "right"
            leftThumb -> "left_thumb"
            rightThumb -> "right_thumb"
            else -> "unknown"
        }
        // Build the request body
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                imageFile.name,
                RequestBody.create("image/png".toMediaTypeOrNull(), imageFile)
            )
            .addFormDataPart("mask", activeHand)
            .build()

        // Build the request
        val request = Request.Builder()
            .url(ngrokUrl + "get_fingerprints")
            .post(requestBody)
            .build()

        // Execute the request using OkHttp
        val client = OkHttpClient.Builder()
            .connectTimeout(90, java.util.concurrent.TimeUnit.SECONDS) // Increase connection timeout
            .writeTimeout(90, java.util.concurrent.TimeUnit.SECONDS)  // Increase write timeout
            .readTimeout(90, java.util.concurrent.TimeUnit.SECONDS)   // Increase read timeout
            .build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(
                        applicationContext,
                        "Upload Failure: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {

                val jsonResponse = response.body?.string()


                if (response.isSuccessful) {
                    try {
                        val jsonObject = jsonResponse?.let { JSONObject(it) }
                        runOnUiThread {
                            if (!isDestroyed && !isFinishing) {
                                val imageUrls = jsonObject?.let { parseJsonImageUrls(it) }
                                Toast.makeText(applicationContext, "Fetching Prints, Please Wait", Toast.LENGTH_LONG).show()
                                showImagesInRecyclerView()
                            }
                        }
                    } catch (e: JSONException) {
                        Log.e("JSON Parse Error", "Failed to parse JSON: ${e.message}")
                        runOnUiThread {
                            Toast.makeText(applicationContext, "Unexpected server response", Toast.LENGTH_LONG).show()
                        }
                    }
                } else {
                    runOnUiThread {
                        try {
                            val jsonObject = jsonResponse?.let { JSONObject(it) }
                            val message = jsonObject?.optString("message", "Unknown error occurred")
                            Toast.makeText(applicationContext, message, Toast.LENGTH_LONG).show()
                        } catch (e: JSONException) {
                            Log.e("JSON Parse Error", "Failed to parse error JSON: ${e.message}")
                            Toast.makeText(applicationContext, "Error: ${jsonResponse ?: "Unknown error"}", Toast.LENGTH_LONG).show()
                        }
                        LoadScanPage()
                    }
                }

            }
        })
    }

    private fun LoadScanPage()
    {
        val intent = Intent(this, ScanPrints::class.java)
        finish() // Destroy the current activity
        startActivity(intent)
    }

    private fun stopCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showProcessingState(message: String) {
        runOnUiThread {
            val spinner = findViewById<ProgressBar>(R.id.progress_spinner)
            val progressMessage = findViewById<TextView>(R.id.progress_message)

            spinner.visibility = View.VISIBLE
            spinner.indeterminateDrawable.setTint(ContextCompat.getColor(this, R.color.dark_blue))

            progressMessage.visibility = View.VISIBLE
            progressMessage.text = message
            progressMessage.setTextColor(ContextCompat.getColor(this, R.color.white))

            if (message == "Processing Complete" || message == "Error capturing image.")
            {
                if(message == "Processing Complete")
                {
                    progressMessage.setTextColor(ContextCompat.getColor(this, R.color.green))
                    spinner.visibility = View.INVISIBLE
                }
                else
                {
                    progressMessage.setTextColor(ContextCompat.getColor(this, R.color.red))
                    spinner.visibility = View.GONE
                }
            }
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        autoCaptureHandler?.removeCallbacksAndMessages(null)
    }
}

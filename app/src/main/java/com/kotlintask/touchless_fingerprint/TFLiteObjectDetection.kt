package com.kotlintask.touchless_fingerprint

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF

import org.opencv.core.Point
import org.opencv.core.MatOfFloat

import org.opencv.core.Rect
import android.util.Log
import android.view.View
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfRect
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect2d
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Dnn.NMSBoxes
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class BoundingBox(// Top-left x-coordinate
    var x1: Float, // Top-left y-coordinate
    var y1: Float, // Bottom-right x-coordinate
    var x2: Float, // Bottom-right y-coordinate
    var y2: Float, // Confidence score
    var cnf: Float, // Class index
    var cls: Int, // Class name
    var clsName: String
) {
    var cx: Float = (x1 + x2) / 2 // Center x-coordinate

    // Compute center x
    var cy: Float = (y1 + y2) / 2 // Center y-coordinate

    // Compute center y
    var w: Float = x2 - x1 // Width of the bounding box

    // Compute width
    var h: Float = y2 - y1 // Height of the bounding box
    // Compute height

    override fun toString(): String {
        return "BoundingBox{" +
                "x1=" + x1 +
                ", y1=" + y1 +
                ", x2=" + x2 +
                ", y2=" + y2 +
                ", cx=" + cx +
                ", cy=" + cy +
                ", w=" + w +
                ", h=" + h +
                ", confidence=" + cnf +
                ", classIndex=" + cls +
                ", className='" + clsName + '\'' +
                '}'
    }
}


class TFLiteObjectDetection(
    modelPath: String,
    labelsPath: String,
    assetManager: AssetManager,

) {
    private val interpreter: Interpreter
    private val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int
    private val numDetections: Int
    val fingersToCheck = listOf("Index-Finger", "Little-Finger", "Middle-Finger", "Ring-Finger")

    val classNames = listOf("Index-Finger","Left-Hand","Little-Finger","Middle-Finger","Right-Hand","Ring-Finger")

    val thumb_classNames = listOf ("left", "left-thumb", "right", "right-thumb")

    val thumbs_to_check=listOf ("left-thumb", "right-thumb")
    private var screenWidth: Int=0
    private var screenHeight: Int=0
    init {
        // Load the TFLite model
        interpreter = Interpreter(loadModelFile(modelPath, assetManager))

        labels = loadLabels(labelsPath, assetManager)

        // Get model input dimensions
        val inputShape = interpreter.getInputTensor(0).shape()
        inputWidth = inputShape[1]
        inputHeight = inputShape[2]
        numDetections = interpreter.getOutputTensor(0).shape()[1] // Number of detections
    }


    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String, assetManager: AssetManager): MappedByteBuffer {
        FileInputStream(assetManager.openFd(modelPath).fileDescriptor).channel.use { fileChannel ->
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                assetManager.openFd(modelPath).startOffset,
                assetManager.openFd(modelPath).declaredLength
            )
        }
    }

    @Throws(IOException::class)
    private fun loadLabels(labelsPath: String, assetManager: AssetManager): List<String> {
        val labels: MutableList<String> = ArrayList()
        try {
            assetManager.open(labelsPath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        line?.let {
                            labels.add(it.trim())
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("loadLabels", "Error loading labels: ${e.message}", e)
            throw e // Re-throw the exception if necessary
        }
        return labels
    }

    fun runInference(bitmap: Bitmap?, isThumb:Boolean,maskwidth:Int,maskheight:Int,leftHand:Boolean):  Pair<Int, MutableList<RectF>> {
        this.screenWidth=maskwidth
        this.screenHeight=maskheight
        // Preprocess the input image
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap!!, inputWidth, inputHeight, true)
        val inputBuffer = preprocessImage(resizedBitmap)
        lateinit var outputArray:Array<Array<FloatArray>>
        if (isThumb)
        {
            outputArray = Array(1) {
                Array(8) {
                    FloatArray(8400)
                }
            }
        }
        else{
            outputArray = Array(1) {
                Array(10) {
                    FloatArray(8400)
                }
            }
        }


        interpreter.run(inputBuffer, outputArray)

        if (isThumb)
        {

            val outputs=outputArray[0]

            return get_results_for_thumb(outputs)
        }
        val (result, boundingBoxes)=processDetectionResults(outputArray[0], resizedBitmap,leftHand)
        return Pair(result,boundingBoxes)

    }

    fun convertArrayToMat(array: Array<FloatArray>): Mat {
        // Get the number of rows and columns from the array
        val rows = array.size
        val cols = array[0].size

        // Create a Mat with the same size and type as the input array
        val mat = Mat(rows, cols, CvType.CV_32F)

        // Fill the Mat with data from the array
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                mat.put(i, j, array[i][j].toDouble()) // Store as a double value
            }
        }

        return mat
    }

    private fun get_results_for_thumb(arrayoutput: Array<FloatArray>): Pair<Int, MutableList<RectF>>
    {

        val boundingBoxes = mutableListOf<RectF>()
        val output= convertArrayToMat(arrayoutput)
        val confidenceThreshold=0.6f
        val iouThreshold=0.2f
        // Transpose the matrix (output.T in Python)
        val transposedOutput = Mat()
        Core.transpose(output, transposedOutput)

        // Extract boxes and scores (equivalent to slicing in NumPy)
        val boxesXYWH = transposedOutput.colRange(0, 4) // Get first 4 columns for boxes (x, y, width, height)
        val scoresMat = transposedOutput.colRange(4, transposedOutput.cols())

        // Convert to float arrays for further processing
        val boxes = ArrayList<FloatArray>()
        for (i in 0 until boxesXYWH.rows()) {
            val box = FloatArray(4)
            for (j in 0 until 4) {
                box[j] = boxesXYWH.get(i, j)[0].toFloat()
            }
            boxes.add(box)
        }

        val scores = FloatArray(scoresMat.rows())
        val classes = IntArray(scoresMat.rows())

        // Find the max score per row and the corresponding class index
        for (i in 0 until scoresMat.rows()) {
            val row = scoresMat.row(i)
            val maxScoreIndex = Core.minMaxLoc(row).maxLoc.x.toInt()
            scores[i] = row.get(0, maxScoreIndex)[0].toFloat()
            classes[i] = maxScoreIndex
        }
//        Log.e("boxes_xywh.shape:","${boxes.size} x 4")



        val matOfRect2d = convertBoxesToMatOfRect2d(boxes)
        val matOfFloat = convertScoresToMatOfFloat(scores)

        // Perform Non-Maximum Suppression (NMS)
        var classesSizeMap:HashMap<String, List<Int>>
                = HashMap<String, List<Int>> ()
        classesSizeMap.put("left", listOf(0,0))
        classesSizeMap.put("left-thumb",listOf(0,0))
        classesSizeMap.put("right",listOf(0,0))
        classesSizeMap.put("right-thumb",listOf(0,0))


        val nmsIndices = MatOfInt()
        val rows = transposedOutput.rows()
        val cols = transposedOutput.cols()

        val classes_index = IntArray(rows)

        for (i in 0 until rows) {
            // Slice the row from index 4 onward (5th column to end)
            val rowSlice = transposedOutput.row(i).colRange(4, cols)

            // Find the index of the maximum value in the sliced row
            val minMax = Core.minMaxLoc(rowSlice)
            classes_index[i] = minMax.maxLoc.x.toInt() + 4 // Adding 4 to shift back the index
        }

//            Log.e("checking: ",classes_index.toString())
        NMSBoxes(matOfRect2d, matOfFloat, confidenceThreshold, iouThreshold, nmsIndices)
        if (nmsIndices.rows()>0){
            for (i in nmsIndices.toList()) {
                if (scores[i] >= 0.5) {
                    val (x_center, y_center, width, height) = boxes[i]

                    // Check if the coordinates are normalized (0.0 to 1.0)
                    val normalized = x_center in 0.0..1.0 && y_center in 0.0..1.0

                    // Image dimensions
                    val imageWidth = 640
                    val imageHeight = 640

                    // If normalized, scale the values
                    val scaledXCenter = if (normalized) x_center * imageWidth else x_center
                    val scaledYCenter = if (normalized) y_center * imageHeight else y_center
                    val scaledWidth = if (normalized) width * imageWidth else width
                    val scaledHeight = if (normalized) height * imageHeight else height

                    // Calculate top-left and bottom-right corners
                    var x1 = ((scaledXCenter - scaledWidth / 2)).toInt()
                    var y1 = ((scaledYCenter - scaledHeight / 2)).toInt()
                    var x2 = ((scaledXCenter + scaledWidth / 2)).toInt()
                    var y2 = ((scaledYCenter + scaledHeight / 2)).toInt()

                    // Clamp coordinates to image bounds
                    x1 = x1.coerceIn(0, imageWidth - 1)
                    y1 = y1.coerceIn(0, imageHeight - 1)
                    x2 = x2.coerceIn(0, imageWidth - 1)
                    y2 = y2.coerceIn(0, imageHeight - 1)
                    boundingBoxes.add(RectF(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat()))
                    // Add the bounding box size to the map
                    classesSizeMap[thumb_classNames[classes_index[i] - 4]] = listOf(x1,y1)
                }
            }

//            Log.e("checking: ",classesSizeMap.get("left").toString()+" "+classesSizeMap.get("right")+"")
            return Pair(checkThumbs(classesSizeMap),boundingBoxes)


        }
//        -5 means no thumb detected!!
        return Pair(-10,boundingBoxes);
    }

    private fun checkThumbs(classesSizeMap: MutableMap<String, List<Int>>):Int
    {
        val result = when {
            // If all values are 0, return -10 (no thumb detected)
            thumbs_to_check.all { thumb ->
                classesSizeMap[thumb]?.all { it == 0 } == true
            } -> -10

            // If index 0 is within ±40 of windowWidth and index 1 is within ±40 of windowHeight, return 100 which means thumb is accurate
            thumbs_to_check.any { thumb ->
                val sizeArray = classesSizeMap[thumb]
                sizeArray != null &&
                        sizeArray[0] in (screenWidth/4 - 70)..(screenWidth/4 + 70) &&
                        sizeArray[1] in (screenHeight/10 - 70)..(screenHeight/10 + 70)
            } -> 100

            // Default case: return -100 means wrong thumb position
            else -> -100
        }
        Log.e("classes 1:","width: "+screenWidth/4+" height: "+screenHeight/10+" result: "+result+" "+classesSizeMap.entries.joinToString(", ") { "${it.key}: ${it.value}" })

        return  result;
//        return -1000
    }

    private fun processDetectionResults(arrayoutput: Array<FloatArray>, bitmap: Bitmap,leftHand:Boolean):  Pair<Int, MutableList<RectF>> {
        val output= convertArrayToMat(arrayoutput)
        println("output shape: ${output.rows()} x ${output.cols()}")
        val confidenceThreshold=0.8f
        val iouThreshold=0.2f
        // Transpose the matrix (output.T in Python)
        val transposedOutput = Mat()
        Core.transpose(output, transposedOutput)

        // Extract boxes and scores (equivalent to slicing in NumPy)
        val boxesXYWH = transposedOutput.colRange(0, 4) // Get first 4 columns for boxes (x, y, width, height)
        val scoresMat = transposedOutput.colRange(4, transposedOutput.cols())

        // Convert to float arrays for further processing
        val boxes = ArrayList<FloatArray>()
        for (i in 0 until boxesXYWH.rows()) {
            val box = FloatArray(4)
            for (j in 0 until 4) {
                box[j] = boxesXYWH.get(i, j)[0].toFloat()
            }
            boxes.add(box)
        }

        val scores = FloatArray(scoresMat.rows())
        val classes = IntArray(scoresMat.rows())

        // Find the max score per row and the corresponding class index
        for (i in 0 until scoresMat.rows()) {
            val row = scoresMat.row(i)
            val maxScoreIndex = Core.minMaxLoc(row).maxLoc.x.toInt()
            scores[i] = row.get(0, maxScoreIndex)[0].toFloat()
            classes[i] = maxScoreIndex
        }
//        Log.e("boxes_xywh.shape:","${boxes.size} x 4")



        val matOfRect2d = convertBoxesToMatOfRect2d(boxes)
        val matOfFloat = convertScoresToMatOfFloat(scores)

       // Perform Non-Maximum Suppression (NMS)
        var classesSizeMap:HashMap<String, List<Int>>
        = HashMap<String, List<Int>> ()
        classesSizeMap.put("Index-Finger", listOf(0,0))
        classesSizeMap.put("Left-Hand",listOf(0,0))
        classesSizeMap.put("Little-Finger",listOf(0,0))
        classesSizeMap.put("Middle-Finger",listOf(0,0))
        classesSizeMap.put("Middle-Finger",listOf(0,0))
        classesSizeMap.put("Ring-Finger",listOf(0,0))


        val nmsIndices = MatOfInt()
        val rows = transposedOutput.rows()
        val cols = transposedOutput.cols()

        val classes_index = IntArray(rows)

        for (i in 0 until rows) {
            // Slice the row from index 4 onward (5th column to end)
            val rowSlice = transposedOutput.row(i).colRange(4, cols)

            // Find the index of the maximum value in the sliced row
            val minMax = Core.minMaxLoc(rowSlice)
            classes_index[i] = minMax.maxLoc.x.toInt() + 4 // Adding 4 to shift back the index
        }

        val boundingBoxes = mutableListOf<RectF>()
        NMSBoxes(matOfRect2d, matOfFloat, confidenceThreshold, iouThreshold, nmsIndices)
        if (nmsIndices.rows()>0){
            for (i in nmsIndices.toList()) {
                if (scores[i] >= 0.5) {
                    val (x_center, y_center, width, height) = boxes[i]

                    val x1 = ((x_center - width / 2) * 640).toInt()
                    val y1 = ((y_center - height / 2) * 640).toInt()
                    val x2 = ((x_center + width / 2) * 640).toInt()
                    val y2 = ((y_center + height / 2) * 640).toInt()
                    if (classNames[classes_index[i]-4]!="Right-Hand" && classNames[classes_index[i]-4]!="Left-Hand")
                    {
                        val x1_bound = ((x_center - width / 2) * 1100).toInt()
                        val y1_bound = ((y_center - height / 2) * 2200).toInt()
                        val x2_bound = ((x_center + width / 2) * 1100).toInt()
                        val y2_bound = ((y_center + height / 2) * 2200).toInt()
                        boundingBoxes.add(RectF(x1_bound.toFloat(), y1_bound.toFloat(), x2_bound.toFloat(), y2_bound.toFloat()))
                    }
                    classesSizeMap[classNames[classes_index[i]-4]] = listOf(x2 - x1,x2)
//                   var ar= classNames[classes_index[i]]
                }
            }
            Log.e("classes 1:",classesSizeMap.entries.joinToString(", ") { "${it.key}: ${it.value}" })

            return Pair(checkFingers(classesSizeMap,leftHand), boundingBoxes)


        }
//    if all conditions are met then 1 should be returned if all value!=null but any of the value is below 110 then return 0 (which means hand is far)
//    and if any value is null return -1 (which mean not all fingers are shown) and -2 if no hand detected
        return Pair(-2, boundingBoxes)

    }

    fun checkFingers(classesSizeMap: MutableMap<String, List<Int>>,leftHand:Boolean): Int {
        // List of fingers to check

//        Log.d("scores",""+classesSizeMap.get("Index-Finger")+" "+classesSizeMap.get("Little-Finger")+" "+classesSizeMap.get("Middle-Finger")+" "+classesSizeMap.get("Ring-Finger"))
        // Check if all conditions are met: each finger is in the map and its value > 105
        // -50 is for the case when the hand orientation is wrong
        if((leftHand && classesSizeMap["Middle-Finger"]?.get(1)!! <400) || (!leftHand && classesSizeMap["Middle-Finger"]?.get(1)!! >400))
        {
            return -50
        }


        val result = when {
            // If any value is null, return -1
            fingersToCheck.any { finger -> classesSizeMap[finger]?.get(0) == 0 } -> -1

            // If all values are greater than 110, return 1
            fingersToCheck.all { finger -> classesSizeMap[finger]?.get(0)?.let { it > 110 && it < 180 } == true } -> 1

            // If any value is less than or equal to 110, return 0
            else -> 0
        }
        return result
    }


    fun convertBoxesToMatOfRect2d(boxes: ArrayList<FloatArray>): MatOfRect2d {
        val matOfRect2d = MatOfRect2d()
        val rects = ArrayList<Rect2d>()

        for (box in boxes) {
            val rect = Rect2d(box[0].toDouble(), box[1].toDouble(), box[2].toDouble(),
                box[3].toDouble()
            ) // x, y, width, height
            rects.add(rect)
        }

        matOfRect2d.fromList(rects)
        return matOfRect2d
    }

    fun convertScoresToMatOfFloat(scores: FloatArray): MatOfFloat {
        val matOfFloat = MatOfFloat()
        matOfFloat.fromArray(*scores) // Convert array to MatOfFloat
        return matOfFloat
    }

    fun drawBoundingBoxes(
        originalBitmap: Bitmap,
        BoundingBoxes: List<BoundingBox>
    ): Bitmap {
        // Create a mutable copy of the original bitmap to draw on
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

        // Create a Canvas to draw on the bitmap
        val canvas = Canvas(mutableBitmap)

        // Define Paint for bounding boxes
        val boxPaint = Paint()
        boxPaint.color = Color.RED // Bounding box color
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 8f

        // Define Paint for text
        val textPaint = Paint()
        textPaint.color = Color.WHITE // Text color

        //        textPaint.setTextSize(40f);
//        textPaint.setTypeface(Typeface.DEFAULT_BOLD);

        // Loop through each bounding box and draw it on the canvas

        for (box in BoundingBoxes) {
            // Scale the bounding box coordinates to match the original image dimensions
            val x1: Float = box.x1 * mutableBitmap.width
            val y1: Float = box.y1 * mutableBitmap.height
            val x2: Float = box.x2 * mutableBitmap.width
            val y2: Float = box.y2 * mutableBitmap.height

            // Draw the bounding box as a rectangle
            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            // Draw the class name and confidence score
            val label: String = box.clsName + " (" + String.format("%.2f", box.cnf) + ")"
            canvas.drawText(label, x1, y1 - 10, textPaint) // Position text slightly above the box
        }

        // Return the modified bitmap
        return mutableBitmap
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4) // RGB
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f // Normalize Red
            val g = ((pixel shr 8) and 0xFF) / 255.0f // Normalize Green
            val b = (pixel and 0xFF) / 255.0f // Normalize Blue
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }

        return buffer
    }

    companion object {
        fun flattenArray(input: Array<Array<FloatArray>>): FloatArray {
            val size = input.size * input[0].size * input[0][0].size
            val flatArray = FloatArray(size)
            var index = 0
            for (twoD in input) {
                for (oneD in twoD) {
                    for (value in oneD) {
                        flatArray[index++] = value
                    }
                }
            }
            return flatArray
        }
    }
}

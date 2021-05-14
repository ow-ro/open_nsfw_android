package io.github.devzwy.nsfw

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.util.*


object NSFWHelper {

    private var nsfwApplication: Context? = null
    private lateinit var mInterpreter: Interpreter
    private val INPUT_WIDTH = 224
    private val INPUT_HEIGHT = 224
    private var isEnableLog = false

    /**
     * The NSFW initialization function Internal log is turned off by default, and the debug environment can open the log using openDebugLog().
     * [application] It is recommended to pass in the application, and there may be a memory leak incoming activity
     * [modelPath] Model file path, which is read from under Assets by default when empty
     * [isOpenGPU] Whether to turn on GPU scan acceleration, some models are compatible with unfriendly can be turned off. Turn on by default
     * [numThreads] Threads internally assigned when scanning data. Default 4
     */

    fun initHelper(
        context: Context,
        modelPath: String? = null,
        isOpenGPU: Boolean = true,
        numThreads: Int = 4
    ) {

        nsfwApplication?.let {

            logD("NSFWHelper initialized, automatically skip this initialization!")

            return
        }


        nsfwApplication = context

        getInterpreterOptions(isOpenGPU, numThreads).let { options ->

            if (modelPath.isNullOrEmpty()) {
                logD("The model path is not passed in, and an attempt is made to read the 'nsfw.tflite' model file from under Assets")
                // When the specified model is empty, look for a model with the name nsfw.tflite in the assets directory by default
                try {

                    mInterpreter = Interpreter(
                        nsfwApplication!!.assets.openFd("nsfw.tflite")
                            .let { fileDescriptor ->
                                FileInputStream(fileDescriptor.fileDescriptor).channel.map(
                                    FileChannel.MapMode.READ_ONLY,
                                    fileDescriptor.startOffset,
                                    fileDescriptor.declaredLength
                                )
                            }, options
                    )
                } catch (mFileNotFoundException: FileNotFoundException) {

                    nsfwApplication = null

                    logE("The 'nsfw.tflite' model was not successfully read from under Assets")

                    throw NSFWException("The 'nsfw.tflite' model was not successfully read from under Assets")
                }

                logD("The model file was loaded successfully from Assets!")

            } else {

                logD("Try reading the model from the incoming model path")

                // Look for the model file under the specified path for initialization
                try {
                    modelPath.let {
                        File(it).let { modelFile ->
                            modelFile.exists().assetBoolean({
                                mInterpreter = Interpreter(
                                    modelFile,
                                    options
                                )
                            }, {
                                throw FileNotFoundException("The model file was not found")
                            })
                        }
                    }

                    logD("The model loaded successfully!")

                } catch (e: Exception) {

                    nsfwApplication = null

                    logE("The model was misconfigured and the read failed")
                    throw NSFWException("The model file was not read correctly: '${modelPath}'")
                }
            }

        }

        logD("NSFWHelper initialization success! ${if (isOpenGPU) "GPU acceleration has been successfully turned on" else "GPU acceleration is not turned on"}")

    }

    /**
     * Turn on the log
     */
    fun openDebugLog() {
        isEnableLog = true
    }

    private fun logD(content: String) {
        if (isEnableLog) Log.d(javaClass.name, content)
    }

    private fun logE(content: String) {
        if (isEnableLog) Log.e(javaClass.name, content)
    }


    private fun getInterpreterOptions(openGPU: Boolean, numThreads: Int): Interpreter.Options {
        return Interpreter.Options().also {
            it.setNumThreads(numThreads)
            if (openGPU) {
                it.addDelegate(GpuDelegate())
                /*CPU go to GPU can read or write data directly to the hardware buffer in the GPU and bypass avoidable memory copies*/
                it.setAllowBufferHandleOutput(true)
                /*CPU-to-GPU processing increases scan speed*/
                it.setAllowFp16PrecisionForFp32(true)
            }
        }
    }

    /**
     * Synchronize scan bitmap
     */
    fun getNSFWScore(bitmap: Bitmap): NSFWScoreBean {

        nsfwApplication?.let {
            SystemClock.uptimeMillis().let { startTime ->
                // Whether dual linear filtering should be used when zooming bitmaps. If this is correct, double linear
                // filtering is used when scaling, resulting in better image quality at the expense of poor performance.
                // If this is wrong, use nearest neighbor scaling, which will make the image less quality but faster. The
                // recommended default setting is to set the filter to True because the cost of a dual linear filter is usually
                // small and improved image quality is important
                ByteArrayOutputStream().let { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
                    stream.close()
                    convertBitmapToByteBuffer(bitmap).let { result ->
                        // out
                        Array(1) { FloatArray(2) }.apply {
                            synchronized(this@NSFWHelper) {
                                mInterpreter.run(result.imgData, this)

                                // Force English for numerical formatting to avoid comma decimals
                                DecimalFormat("0.000", DecimalFormatSymbols(Locale.ENGLISH)).let {
                                    return NSFWScoreBean(
                                        it.format(this[0][1]).toFloat(),
                                        it.format(this[0][0]).toFloat(),
                                        result.exceTime,
                                        SystemClock.uptimeMillis() - startTime
                                    ).also {
                                        logD("The scan is complete: (${result}) -> $it")
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }
        throw NSFWException("Call NSFWHelper.init (...) Try again!")

    }

    /**
     * Asynchronous scan file NSFW value
     */
    fun getNSFWScore(bitmap: Bitmap, onResult: ((NSFWScoreBean) -> Unit)) {
        // Get a picture image.open (path) by path.
        // Determine whether the pixel format of the picture is RGB, and convert to RGB if not RGB (24-bit color
        // image, each represented by 24 bits, representing three channels of red, green, and blue).
        // Resize the picture to a size of 256 x 256 and use the Official Explanation: To resize, use linear interpolation
        // to calculate the output pixel value for all pixels that may affect the output value. For other transformations,
        // use linear interpolation on the 2x2 environment in the input image.
        // The results of resize are converted to an io stream and saved in JPEG format.
        // Convert to 64-bit floating point.asType float32 defines variable storage for converted 32-bit float picture data.
        // Get the width and height of the picture, intercept 224 x 224 size x start at 16 bits, and take to 16 plus 224 position y as well.
        // The value conversion bit float32 will be taken.
        // Each color value will be * 255.
        // Subtract each color by a certain threshold of 104...
        // [[[127.64.-18]]] Convert to [[[[127.64.-18]]]]
        // Feed the model using the index keyword.
        // Delete all single-dimensional entries.
        // Output scan results.

        GlobalScope.launch(Dispatchers.IO) {
            getNSFWScore(bitmap).let { result ->
                withContext(Dispatchers.Main) {
                    onResult(result)
                }
            }
        }
    }

    /**
     * Load the scan data
     */
    private fun convertBitmapToByteBuffer(bitmap_: Bitmap): CovertBitmapResultBean {

        ByteBuffer.allocateDirect(1 * INPUT_WIDTH * INPUT_HEIGHT * 3 * 4).let { imgData ->

            imgData.order(ByteOrder.LITTLE_ENDIAN)

            SystemClock.uptimeMillis().let { startTime ->
                imgData.rewind()
                IntArray(INPUT_WIDTH * INPUT_HEIGHT).let {
                    // Convert the color value of each pixel to intValues
                    bitmap_.getPixels(
                        it,
                        0,
                        INPUT_WIDTH,
                        Math.max((bitmap_.height - INPUT_HEIGHT) / 2, 0),
                        Math.max((bitmap_.width - INPUT_WIDTH) / 2, 0),
                        INPUT_WIDTH,
                        INPUT_HEIGHT
                    ).let {
                        bitmap_.recycle()
                    }
                    for (color in it) {
                        imgData.putFloat((Color.blue(color) - 104).toFloat())
                        imgData.putFloat((Color.green(color) - 117).toFloat())
                        imgData.putFloat((Color.red(color) - 123).toFloat())
                    }
                }
                return CovertBitmapResultBean(imgData, SystemClock.uptimeMillis() - startTime)
            }
        }

    }

}
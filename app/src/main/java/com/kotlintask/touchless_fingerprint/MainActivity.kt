package com.kotlintask.touchless_fingerprint

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import androidx.core.content.ContextCompat
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts

class MainActivity : AppCompatActivity()
{
    private lateinit var btn: Button
    override fun onCreate(savedInstanceState: Bundle?)
    {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btn = findViewById(R.id.Scan_Prints_Button)

        // Check to see if OpenCV is loaded successfully
        if (OpenCVLoader.initLocal())
        {
            Log.d("OpenCV", "OpenCV loaded successfully")
        }
        else
        {
            Log.e("OpenCV", "OpenCV failed to load")
        }

        if (hasCameraPermission())
        {
            btn.setOnClickListener{
                val intent = Intent(this, ScanPrints::class.java)
                startActivity(intent)
            }
        }
        else
        {
            requestCameraPermission()
        }
    }

    private fun hasCameraPermission(): Boolean
    {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission()
    {
        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
                if (isGranted) {
                    // Permission granted, go to scan
                    btn.setOnClickListener {
                        // intent to new page
                        Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
                        val intent = Intent(this, ScanPrints::class.java)
                        startActivity(intent)
                    }

                } else {
                    // Permission denied, show a message
                    Toast.makeText(this, "Camera permission is required to use the app", Toast.LENGTH_SHORT).show()
                }
            }
        requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

}
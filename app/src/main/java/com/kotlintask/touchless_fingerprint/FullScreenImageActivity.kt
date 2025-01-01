package com.kotlintask.touchless_fingerprint

import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.bumptech.glide.Glide

class FullScreenImageActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_full_screen_image)

        val imageUrl = intent.getStringExtra("image_url")
        val imageView = findViewById<ImageView>(R.id.full_screen_image)

        Glide.with(this).load(imageUrl).into(imageView)

        // Handle back button
        val backButton = findViewById<Button>(R.id.back_button)
        backButton.setOnClickListener {
            finish()
        }
    }
}

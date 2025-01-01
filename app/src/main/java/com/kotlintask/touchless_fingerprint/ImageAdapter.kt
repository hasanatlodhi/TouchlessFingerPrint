package com.kotlintask.touchless_fingerprint

import android.content.Intent
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide

class ImageAdapter(
    private val imageUrls: List<String>,
    private val handinfo: String,
    private val onClick: (String) -> Unit
) : RecyclerView.Adapter<ImageAdapter.ImageViewHolder>() {

    class ImageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.image_item)
        val imageButton: ImageView = itemView.findViewById(R.id.back_button_recyclerView)
        val finger_name: TextView=itemView.findViewById(R.id.finger_name)
        val scroll_text:TextView=itemView.findViewById(R.id.scroll_text)
    }
    fun extractFingerName(url: String): String {
        return url.split("/").last()
    }
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.image_item_layout, parent, false)
        return ImageViewHolder(view)
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        val imageUrl = imageUrls[position]
        if (position>0 || handinfo=="Left Thumb" || handinfo=="Right Thumb")
        {
            holder.scroll_text.text=""
        }
        Log.d("ImageAdapter", "Binding image at position $position: $imageUrl")
        var finger_name=""
        if (handinfo=="Left Thumb" || handinfo=="Right Thumb")
        {
            finger_name=handinfo
        }
        else{
            finger_name=extractFingerName(imageUrl).removeSuffix(".png")
        }


        holder.finger_name.setText(finger_name)
        Glide.with(holder.imageView.context)
            .load(imageUrl)
            .into(holder.imageView)

        holder.imageButton.setOnClickListener {
            val intent = Intent(holder.imageButton.context, ScanPrints::class.java)
            holder.imageButton.context.startActivity(intent)
        }

        holder.imageView.setOnClickListener {
            Log.d("ImageAdapter", "Image clicked at position $position")
            onClick(imageUrl)
        }
    }

    override fun getItemCount(): Int {
        Log.d("ImageAdapter", "Item count: ${imageUrls.size}")
        return imageUrls.size
    }
}


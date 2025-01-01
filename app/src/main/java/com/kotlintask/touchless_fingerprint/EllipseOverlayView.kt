package com.kotlintask.touchless_fingerprint

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class EllipseOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paint = Paint().apply {
        color = context.getColor(android.R.color.holo_red_light) // Change as needed
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    var boundingBoxes: List<RectF> = emptyList()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.let {
            for (rect in boundingBoxes) {
                it.drawOval(rect, paint)
            }
        }
    }
}

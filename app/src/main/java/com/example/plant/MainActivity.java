package com.example.plant;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.plant.ml.PlantDiseaseModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
ImageView imageView;
Button gallery;
TextView result;
    ActivityResultLauncher<String> mgetcontent;
    private static final int CAMERA_REQUEST_CODE = 1001;
    private Button captureButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        imageView = findViewById(R.id.image);
        gallery = findViewById(R.id.gallery);
        result = findViewById(R.id.result);
        captureButton = findViewById(R.id.camerabtn);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create an intent to capture an image from the camera
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                // Check if there's a camera app to handle the intent
                if (cameraIntent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(cameraIntent, CAMERA_REQUEST_CODE);
                } else {
                    Toast.makeText(MainActivity.this, "No camera app found", Toast.LENGTH_SHORT).show();
                }
            }
        });
        mgetcontent = registerForActivityResult(new ActivityResultContracts.GetContent(), new ActivityResultCallback<Uri>() {
            @Override
            public void onActivityResult(Uri o) {
                Bitmap imagebitmap = null;
                try {
                    imagebitmap = UriToBitmap(o);
                }catch (IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(imagebitmap);
                OutputGenerate(imagebitmap);
            }


        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mgetcontent.launch("image/*");
            }
        });
    }
    private void OutputGenerate(Bitmap imagebitmap) {
        try {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(imagebitmap, 224, 224, true);

            // Normalize the image to [0, 1] and convert to ByteBuffer
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedBitmap);
            PlantDiseaseModel model = PlantDiseaseModel.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            PlantDiseaseModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences = outputFeature0.getFloatArray();

            // Debugging: Log the output length
            System.out.println("Model output length: " + confidences.length);

            if (confidences == null || confidences.length == 0) {
                result.setText("Invalid model output.");
                model.close();
                return;
            }

            int maxIdx = getMaxIndex(confidences);

            // Update labels array to match your model
            String[] labels = {"apple apple scab",
                    "apple black rot",
                    "apple cedar apple rust",
                    "apple healthy"
                    ,"blueberry healthy"
                    ,"cherry including sour powdery mildew"
                    ,"cherry including sour healthy"
                    ,"corn maize cercospora leaf spot gray leaf spot"
                    ,"corn maize common rust"
                    ,"corn maize northern leaf blight"
                    ,"corn maize healthy"
                    ,"grape black rot"
                    ,"grape esca black measles"
                    ,"grape leaf blight isariopsis leaf spot"
                    ,"grape healthy"
                    ,"orange haunglongbing citrus greening"
                    ,"peach bacterial spot"
                    ,"peach healthy"
                    ,"pepper bell bacterial spot"
                    ,"pepper bell healthy"
                    ,"potato early blight"
                    ,"potato late blight"
                    ,"potato healthy"
                    ,"raspberry healthy"
                    ,"soybean healthy"
                    ,"squash powdery mildew"
                    ,"strawberry leaf scorch"
                    ,"strawberry healthy"
                    ,"tomato bacterial spot"
                    ,"tomato early blight"
                    ,"tomato late blight"
                    ,"tomato leaf mold"
                    ,"tomato septoria leaf spot"
                    ,"tomato spider mites two spotted spider mite"
                    ,"tomato target spot"
                    ,"tomato tomato yellow leaf curl virus"
                    ,"tomato tomato mosaic virus"
                    ,"tomato healthy"};

            if (maxIdx >= labels.length) {
                result.setText("Prediction index out of bounds. Check model and labels.");
                model.close();
                return;
            }

            String prediction = labels[maxIdx];
            result.setText(prediction);
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
            result.setText("Error processing image. Please try again.");
        }

    }
    private Bitmap UriToBitmap(Uri o) throws IOException {
        return MediaStore.Images.Media.getBitmap(this.getContentResolver(),o);
    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixelIndex = 0;
        for (int i = 0; i < 224; i++) {
            for (int j = 0; j < 224; j++) {
                int pixel = intValues[pixelIndex++];
                byteBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // Red
                byteBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // Green
                byteBuffer.putFloat((pixel & 0xFF) / 255.0f);         // Blue
            }
        }
        return byteBuffer;
    }
    private int getMaxIndex(float[] confidences) {
        int maxIdx = 0;
        float maxConfidence = confidences[0];
        if (confidences == null || confidences.length == 0) {
            result.setText("Invalid model output.");

        }
        for (int i = 1; i < confidences.length; i++) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // Check if the request code matches and the result is okay
        if (requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK) {
            // Get the captured image from the camera intent
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");

            // Set the captured image to the ImageView
            imageView.setImageBitmap(imageBitmap);
        }
    }
}
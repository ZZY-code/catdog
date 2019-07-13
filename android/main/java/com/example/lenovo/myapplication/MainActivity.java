package com.example.lenovo.myapplication;
import com.example.lenovo.myapplication.R;
import android.os.Build;
import android.os.Bundle;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import java.io.File;
import android.util.Log;
import android.widget.TextView;
import android.os.AsyncTask;
import android.util.Log;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.io.InputStream;

public class MainActivity extends Activity {

    static {
        System.loadLibrary("tensorflow_inference");
        Log.i("wumei","load tensorflow_inference successfully");
    }

    private static final int TAKE_PHOTO = 11;// 拍照
    private static final int CROP_PHOTO = 12;// 裁剪图片
    private static final int LOCAL_CROP = 13;// 本地图库
    private String MODEL_PATH = "file:///android_asset/model.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //保存图片和图片尺寸的
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {128,128,3};

    Button btn_choose_picture;
    private ImageView iv_show_picture;
    private TextView result;
    private Uri imageUri;// 拍照时的图片uri


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setViews();// 初始化控件
        setListeners();// 设置监听
    }


    /**
     * 设置监听
     */
    private void setListeners() {

        // 展示图片按钮点击事件
        btn_choose_picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                takePhotoOrSelectPicture();// 拍照或者调用图库

            }
        });

    }

    private void takePhotoOrSelectPicture() {
        CharSequence[] items = {"拍照", "图库"};// 裁剪items选项

        // 弹出对话框提示用户拍照或者是通过本地图库选择图片
        new AlertDialog.Builder(MainActivity.this)
                .setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                        switch (which) {
                            // 选择了拍照
                            case 0:
                                // 创建文件保存拍照的图片

                                File takePhotoImage = new File(Environment.getExternalStorageDirectory(), "take_photo_image2.jpg");
                                try {
                                    // 文件存在，删除文件
                                    if (takePhotoImage.exists()) {
                                        takePhotoImage.delete();
                                    }
                                    // 根据路径名自动的创建一个新的空文件
                                    takePhotoImage.createNewFile();
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }

                                // 获取图片文件的uri对象
                                imageUri = Uri.fromFile(takePhotoImage);
                                // 创建Intent，用于启动手机的照相机拍照

                                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                                // 指定输出到文件uri中
                                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);

                                // 启动intent开始拍照
                                startActivityForResult(intent, TAKE_PHOTO);
                                break;
                            // 调用系统图库
                            case 1:

                                // 创建Intent，用于打开手机本地图库选择图片
                                Intent intent1 = new Intent(Intent.ACTION_PICK,
                                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                                // 启动intent打开本地图库
                                startActivityForResult(intent1, LOCAL_CROP);
                                break;

                        }

                    }
                }).show();
    }


    /**
     * 调用startActivityForResult方法启动一个intent后，
     * 可以在该方法中拿到返回的数据
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        switch (requestCode) {

            case TAKE_PHOTO:// 拍照

                if (resultCode == RESULT_OK) {
                    // 创建intent用于裁剪图片
                    Intent intent = new Intent("com.android.camera.action.CROP");
                    // 设置数据为文件uri，类型为图片格式
                    intent.setDataAndType(imageUri, "image/*");
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        StrictMode.VmPolicy.Builder builder = new StrictMode.VmPolicy.Builder();
                        StrictMode.setVmPolicy(builder.build());
                    }
                    //  设置裁剪图片的宽高
                    intent.putExtra("outputX", 300);
                    intent.putExtra("outputY", 300);
                    // 允许裁剪
                    intent.putExtra("scale", true);
                    // 指定输出到文件uri中
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                    intent.putExtra("return-data", true);
                    // 启动intent，开始裁剪
                    startActivityForResult(intent, CROP_PHOTO);
                }
                break;
            case LOCAL_CROP:// 系统图库

                if (resultCode == RESULT_OK) {
                    // 创建intent用于裁剪图片
                    Intent intent1 = new Intent("com.android.camera.action.CROP");
                    // 获取图库所选图片的uri
                    Uri uri = data.getData();
                    intent1.setDataAndType(uri, "image/*");
                    //  设置裁剪图片的宽高
                    intent1.putExtra("outputX", 300);
                    intent1.putExtra("outputY", 300);
                    // 裁剪后返回数据
                    intent1.putExtra("return-data", true);
                    // 启动intent，开始裁剪
                    startActivityForResult(intent1, CROP_PHOTO);
                }

                break;
            case CROP_PHOTO:// 裁剪后展示图片
                if (resultCode == RESULT_OK) {
                    try {
                        System.out.print("111111111111");
                        // 展示拍照后裁剪的图片
                        if (imageUri != null) {
                            System.out.print("2222222222222222");
                            // 创建BitmapFactory.Options对象
                            BitmapFactory.Options option = new BitmapFactory.Options();
                            // 属性设置，用于压缩bitmap对象
                            option.inSampleSize = 2;
                            option.inPreferredConfig = Bitmap.Config.RGB_565;
                            // 根据文件流解析生成Bitmap对象
                            Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri), null, option);
                            // 展示图片
                            iv_show_picture.setImageBitmap(bitmap);
                            predict(bitmap);
                        }
                        System.out.print("33333333333333333");
                        // 展示图库中选择裁剪后的图片
                        if (data != null) {
                            System.out.print("444444444444444");
                            // 根据返回的data，获取Bitmap对象
                            Bitmap bitmap = data.getExtras().getParcelable("data");
                            // 展示图片
                            iv_show_picture.setImageBitmap(bitmap);
                            Log.i("1111","CROP_PHOTO");
                            predict(bitmap);
                            Log.i("2222","CROP_PHOTO");
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                break;

        }

    }


    public Object[] argmax(float[] array){

        int best = -1;
        float best_confidence = 0.0f;
        for(int i = 0;i < array.length;i++){
            float value = array[i];
            if (value > best_confidence){
                best_confidence = value;
                best = i;
            }
        }
        return new Object[]{best,best_confidence};
    }


    protected void predict(final Bitmap bitmap){

                tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);
                Bitmap resized_image = ImageUtils.processBitmap(bitmap,128);
                floatValues = ImageUtils.normalizeBitmap(resized_image,128,127.5f,1.0f);
                tf.feed(INPUT_NAME,floatValues,1,128,128,3);

                tf.run(new String[]{OUTPUT_NAME});
                tf.fetch(OUTPUT_NAME,PREDICTIONS);
                Object[] results = argmax(PREDICTIONS);
                int class_index = (Integer) results[0];
                float confidence = (Float) results[1];

                try{
                    final String conf = String.valueOf(confidence * 100).substring(0,5);
                    //Convert predicted class index into actual label name
                    final String label = ImageUtils.getLabel(getAssets().open("label2.json"),class_index);
                    //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Log.i("3333","PREDICT");
                            result.setText("识别结果:"+label ); //这里控制textview显示当前的结果值
                        }
                    });
                } catch (Exception e){
                    Log.i("4444","PREDICT");
                }

    }


    /**
     * 控件初始化
     */
    private void setViews() {
        btn_choose_picture = (Button) findViewById(R.id.btn_choose_picture);
        iv_show_picture = (ImageView) findViewById(R.id.iv_show_picture);
        result = (TextView)findViewById(R.id.result);

    }


}






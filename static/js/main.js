$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    //预览
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // 加载动画
        $(this).hide();
        $('.loader').show();

        // 调用 /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // 得到结果并显示
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('识别结果:  ' + data);
                console.log('Success!');
            },
        });
    });

});

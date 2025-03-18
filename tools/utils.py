# Thêm các thư viện cần thiết
import numpy as numpy
import cv2

# Lật ảnh
def horizontal_flip(image, steering_angle):
    # Lật ảnh theo chiều ngang (Đối xứng trục y)
    flipped_image = cv2.flip(image, 1)
    
    # Đảo ngược góc lái
    steering_angle = -steering_angle
    
    return flipped_image, steering_angle

# Giảm ngẫu nhiên độ sáng cho ảnh
def brightness_reduction(image):
    # Chuyển ảnh từ form màu RGB sang HSV để giảm độ snags
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Chuyển đổi kiểu dữ liệu của pixel ảnh thành số thực cho tính toán dễ dàng 
    image = np.array(image, dtype = np.float64)
    
    # Điều chỉnh độ sáng bằng cách thay đổi thông số V của ảnh
    random_brightness = 1 - np.random.uniform(0.2, 0.4)
    image[:,:,2] = image[:,:,2] * random_brightness
    
    # Chuyển đổi lại kiểu dữ liệu cho pixel của ảnh uint8 
    image = np.array(image, dtype = np.uint8)
    
    # Chuyển ảnh lại snag form màu RGB
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    return image

# Dịch chuyển ảnh ngẫu nhiên theo trục x và trục y
def translation(image, steering_angle, x_trans_range = [-60, 60], y_trans_range = [-20, 20]):
    # Lấy kích thước của ảnh
    height, width = (image.shape[0], image.shape[1])
    
    # Xác đinh độ dịch chuyển của ảnh dọc theo trục x và dọc theo trục y
    x_trans = np.random.randint(x_trans_range[0], x_trans_range[1]) 
    y_trans = np.random.randint(y_trans_range[0], y_trans_range[1])
    
    # Điều chỉnh lại góc lại theo trục x
    steering_angle += x_trans * 0.004
    
    # Tạo ma trận dịch chuyển ảnh (chỉ dịch chuyển đơn thuần theo trục x và y)
    trans_matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
    
    # Dịch chuyển ảnh sử dụng warpAffine
    translated_image = cv2.warpAffine(image, trans_matrix, (width, height))
    
    return translated_image, steering_angle

# Cắt phần vùng bầu trờ của ảnh
def top_bottom_crop(image):
    # Cắt ảnh theo chiều trục y trong khoảng pixel từ 40 - 125
    cropped_image = image[40:135, :]
    
    return cropped_image

# Kết hợp các phương pháp augmention xử lý ảnh
def augment_image(df):
    # Lấy một giá trị ngẫu nhiên trong khoản từ 0 3
    camera_side = np.random.randint(3)

    # Điều chỉnh lại góc tuỳ theo các loại ảnh từ các góc cam
    if camera_side == 0:
        image_path = df.iloc[0]['center_camera'].strip()
        angle_calib = 0
    elif camera_side == 1:
        image_path = df.iloc[0]['left_camera'].strip()
        angle_calib = 0.25
    elif camera_side == 2:
        image_path = df.iloc[0]['right_camera'].strip()
        angle_calib = -0.25

    steering_angle = df.iloc[0]['steering_angle'] + angle_calib

    # Đọc ảnh dùng thư viện opencv
    image = cv2.imread(image_path)

    # Chuyển ảnh từ form BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Ngẫu nhiên augmention ảnh

    if np.random.rand() < 0.5:
        # Dịch chuyển ảnh
        image, steering_angle = translation(image, steering_angle)

    if np.random.rand() < 0.5:
        # Giảm độ sáng của ảnh
        image = brightness_reduction(image)

    if np.random.rand() < 0.5:
        # Lật ảnh
        image, steering_angle = horizontal_flip(image, steering_angle)

    return image, steering_angle

# Tiền xử lý ảnh
def image_preprocessing(image):
    # Cắt ảnh
    image = top_bottom_crop(image)

    # Chuyển ảnh từ form RGB sang YUV cho dễ dàng xử lý
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Resize ảnh
    image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)

    # Scale ảnh
    image = image / 255

    return image

def batch_generator(df, batch_size, training_flag):
    while True:
        # Tạo 1 list lưu ảnh và góc sau khi xử lý từ batch
        images_bacth = []
        steering_angles_batch = []

        for i in range(batch_size):
            # Chọn một hàng ngẫu nhiên với địa chỉ
            index = np.random.randint(0, len(df) - 1)

            # Augmentation lại ảnh cho dữ liệu training
            if training_flag:
                # Augmentation lại ảnh
                image, steering_angle = augment_image(df.iloc[[index]])
            else:
                camera_side = np.random.randint(3)

                # Điều chỉnh lại góc tuỳ theo các loại ảnh từ các góc cam
                if camera_side == 0:
                    image_path = df.iloc[0]['center_camera'].strip()
                    angle_calib = 0
                elif camera_side == 1:
                    image_path = df.iloc[0]['left_camera'].strip()
                    angle_calib = 0.25
                elif camera_side == 2:
                    image_path = df.iloc[0]['right_camera'].strip()
                    angle_calib = -0.25

                # Đọc ảnh tại form RGB
                image = cv2.imread(image_path)
                steering_angle = df.iloc[0]['steering_angle'] + angle_calib

            # Tiền xử lý ảnh
            image = image_preprocessing(image)

            # Add ảnh và góc lái vào các list trên
            images_batch.append(image)
            steering_angles_batch.append(steering_angle)

        yield (np.asarray(images_batch), np.asarray(steering_angles_batch))

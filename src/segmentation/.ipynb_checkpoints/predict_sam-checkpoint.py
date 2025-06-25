from segment_anything import sam_model_registry, SamPredictor
import cv2
import pyhdust.images as phim
import xml.etree.ElementTree as ET


def load_sam():
    sam_checkpoint = "models/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to('cuda')
    predictor = SamPredictor(sam)
    return predictor

def create_balanced_image(temp_img):
     Gray = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
     B = temp_img[:, :, 0]
     G = temp_img[:, :, 1]
     R = temp_img[:, :, 2]

     mean_gray = np.mean(Gray)
     mean_R = np.mean(R)
     mean_G = np.mean(G)
     mean_B = np.mean(B)
     R_ = R * (mean_gray / mean_R)
     G_ = G * (mean_gray / mean_G)
     B_ = B * (mean_gray / mean_B)

     B_[B_>255] = 255
     G_[G_>255] = 255
     R_[R_>255] = 255
     balance_img = temp_img.copy()
     balance_img[:, :, 0] = B_.copy()
     balance_img[:, :, 1] = G_.copy()
     balance_img[:, :, 2] = R_.copy()
     balance_img = cv2.cvtColor(balance_img, cv2.COLOR_BGR2RGB)
     return balance_img

def create_mask_with_km_ms(img_file):
    """
    :param img: input rgb image
    :param min_area: minimum area of nucleus, if area of nucleus is lower than this value, this means
            that the nucleus is not detected
    :return: binary of nucleus, binary of convexhull, binary of ROC
    """
    temp_img = cv2.imread(img_file)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    balance_img = create_balanced_image(temp_img)
    cmyk = phim.rgb2cmyk(balance_img)
    _M = cmyk[:, :, 1]
    _K = cmyk[:, :, 3]

    _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]
    min_MS = np.minimum(_M, _S)
    a_temp = np.where(_K < _M, _K, _M)
    KM = _K - a_temp
    b_temp = np.where(min_MS < KM, min_MS, KM)
    min_MS_KM = min_MS - b_temp
    # cv2.imshow('Step 1' , cv2.resize(Nucleus_img , (256 ,256)))

    # Step 2 :
    min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)
    try:
        thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
        Nucleus_img = np.zeros_like(min_MS_KM)
        Nucleus_img[min_MS_KM >= thresh2] = 255
    except:
        print('try-Except')
        _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
        thresh2 = fl.threshold_multiotsu(_M, 2)
        Nucleus_img = np.zeros_like(_M)
        Nucleus_img[_M >= thresh2] = 255

    contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_del = np.zeros_like(Nucleus_img)

    max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < (max_area / 10):
            cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
    Nucleus_img[pad_del > 0] = 0
    return Nucleus_img

def predict(img_path_tot,bbox_name):
    predictor = load_sam()
    image = cv2.imread(img_path_tot)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(bbox_name, 'r') as f:
        if bbox_name[:-3] == 'txt':
            line = f.readlines()[0]
            
            cell_box = line.strip().split()
            cell_box = [int(i) for i in cell_box]
            label = img_path_tot.split('-')[1]
            
        elif bbox_name[:-3] == 'xml':
            # Parse XML
            tree = ET.parse(bbox_name)
            root = tree.getroot()
            cell_box = []
    
            for obj in root.findall('object'):
                label = obj.find('name').text.split(' ')[0]
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                cell_box.append([xmin, ymin, xmax, ymax])
                
    
    new_cell_box = np.array(cell_box)
    predictor.set_image(image) #array
    mask_cyto, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=new_cell_box[None,:],
    multimask_output=False,
    )
    
    mask_cyto = mask_cyto[0].astype('int32')
    mask_nucleus = create_mask_with_km_ms(images_path_test + 'images/' + img_path)  

    mask_nucleus[mask_cyto == 0] = 0
    
    mask = np.asarray(mask_cyto)
    mask[mask_cyto == 0] = 0
    mask[mask_cyto == 1] = 1
    mask[mask_nucleus == 255] = 2
    return mask, label


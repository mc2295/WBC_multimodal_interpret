from segmentation.

def show_mask(mask, ax, random_color=False, color_name = 1):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if color_name ==1:
            color = np.array([30/255, 255/255, 30/255, 0.6])
        if color_name == 2:
            color = np.array([255/255, 255/255, 0, 0.6])
    h, w = mask.shape[-2], mask.shape[-1]
    
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image, alpha = alpha)
    return mask_image

def create_overlay(mask, ax, title = 'out'):

    maskcyto = np.where(mask ==1, 1, 0)
    masknucleus = np.where(mask ==2, 1, 0)
    
    maskcyto_i = show_mask(maskcyto, ax, color_name = 1)
    masknucleus_i = show_mask(masknucleus,ax, color_name = 2)
    
    masktot = np.zeros_like(masknucleus_i)
    masktot[:,:,0] = maskcyto
    masktot[:,:,1] = maskcyto
    masktot[:,:,2] = maskcyto
    masktot[:,:,3] = maskcyto
    masktot = np.where(masktot ==  1, maskcyto_i, masknucleus_i)
    
    ax.imshow(masktot)
    ax.axis('off')
    plt.savefig('figures/overlay_' + title + '_' + img_path,bbox_inches='tight' )

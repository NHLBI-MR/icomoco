import gadgetron
import numpy as np 
import ismrmrd as mrd
import registration_oflow3D as reg
import time
import sys

def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def create_ismrmrd_image(data, field_of_view, index):
        return mrd.image.Image.from_array(
            np.float32(data),
            image_series_index=index,
            image_type=mrd.IMTYPE_REAL,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=False
        )
        
        
def registration_gadget(connection):
    imgs=[]
    img_hdr=[]
    n=0
    
    params = _parse_params(connection.config)

    if "bidirectional" in params:
        if params["bidirectional"] == 'True':
            bidirectional = True
        else:
            bidirectional = False
    else:
        bidirectional = False  
        
    hdr=connection.header
    field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
    
    #connection.filter(lambda input: type(input) == gadgetron.IsmrmrdImageArray)
    for item in connection:
        if item.data.dtype==np.complex64:
            if item.meta['GADGETRON_DataRole'] == 'Image' and item.image_series_index == 1:
                imgs.append(item.data)    
                # img_hdr.append(item.header)
            connection.send(item)
            
    images = np.transpose(np.concatenate(imgs,axis=0),[0,2,3,1])
    
    if(bidirectional):
        ref_index = int(images.shape[0]/2)
    else:
        ref_index=0
        
    st = time.time()   
    
    registered_images, deformation_fields = reg.register_images(images,ref_index)
    
    eprint("Registration Time: ", time.time()-st)

    for ii in range(0,registered_images.shape[0]):
        
        # image = create_ismrmrd_image(registered_images[ii,...].squeeze(), img_hdr[0], field_of_view, ii)
        # connection.send(image)
        deformation_field = create_ismrmrd_image(np.transpose(deformation_fields[ii,...].squeeze(),[0,3,1,2]), field_of_view, 111)
        connection.send(deformation_field)    

        
if __name__ == "__main__":
    gadgetron.external.listen(21000,registration_gadget)
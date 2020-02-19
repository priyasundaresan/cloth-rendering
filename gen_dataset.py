import os

def gen_data(name, coord_offset, num, noise=False):
    os.system('/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender -b -P rope-blender.py -- --coord_offset=%d --num_images=%d' %(coord_offset, num))
    os.system('python3 mask.py')
    if noise:
        os.system('python3 process_sim.py')
        os.system('mv images/knots_info.json ./images_noisy/')
        os.system('rm -rf images')
        os.system('mv images_noisy ./images')
    os.system('mkdir {}'.format(name))
    os.system('mkdir {}/processed'.format(name))
    os.system('mv images image_masks ./{}/processed/'.format(name))

if __name__ == '__main__':
    names = ['rope_1418_knot_task']
    coord_offsets = [25]
    for i in range(len(names)):
        noise=True
        #gen_data(names[i], coord_offsets[i], 3600, noise=noise)
        gen_data(names[i]+'_test', coord_offsets[i], 10, noise=noise)
    #os.system('rsync -av rope_1418* priya@jensen.ist.berkeley.edu:/raid/priya/data/pdc_synthetic_2/logs_proto/')

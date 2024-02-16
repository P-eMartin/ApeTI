import numpy as np, pdb
from mmpose.datasets import PIPELINES as ppose
from mmdet.datasets import PIPELINES as pdet

@ppose.register_module()
@pdet.register_module()
class LoadTI:
   def __init__(self,
               color_type='temperature',
               channel_order = 'one_channel'):
      self.color_type = color_type
      self.channel_order = channel_order
      
   def __call__(self, results):
      """
      Load np data.
      """
      if 'img_or_path' in results.keys():
         if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            results['img'] = np.float32(np.load(results['image_file'])[..., np.newaxis])
         else:
            results['img'] = np.float32(results['img_or_path'])
            results['image_file'] = ''
      elif 'img_info' in results.keys():
         results['img'] = np.float32(np.load(results['img_info']['file_name'])[..., np.newaxis])
         results['filename'] = results['img_info']['file_name']
         results['ori_filename'] = results['filename']
         results['img_shape'] = results['img'].shape
         results['ori_shape'] = results['img'].shape
      else: # case for keypoints pipeline
         results['img'] = np.float32(np.load(results['image_file'])[..., np.newaxis])
      return results

@ppose.register_module()
@pdet.register_module()
class Debug:
   def __call__(self, results):
      pdb.set_trace()
      return results
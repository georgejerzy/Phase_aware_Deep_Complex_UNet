import os, argparse
from model_module import *

from model import *
from model_loss import *
from model_data import *

from complex_layers.STFT import *
from complex_layers.networks import *
from complex_layers.activations import *


speech_length = 16384
sampling_rate = 16000


'GET UNSEEN SPEECH FILE PATH'
def get_file_list (file_path):
      file_list = []

      for root, dirs, files in (os.walk(file_path)):
            for fname in files: 
                  if fname == "desktop.ini" or fname == ".DS_Store": continue 

                  full_fname = os.path.join(root, fname)
                  file_list.append(full_fname)

      file_list = natsort.natsorted(file_list, reverse = False)
      file_list = np.array(file_list)

      return file_list

import soundfile as sf
'INFERENCE DEEP LEARNING MODEL'
def inference (path_list, save_path):
      sampling_rate = 16_000
      for index1, speech_file_path in tqdm(enumerate (path_list)):
            # _, unseen_noisy_speech = scipy.io.wavfile.read(speech_file_path)

            unseen_noisy_speech, sr = sf.read(speech_file_path, dtype="float32")
            assert sr == sampling_rate

            restore = []
            
            for index2 in range (int(len(unseen_noisy_speech) / speech_length)):
                  split_speech = unseen_noisy_speech[speech_length * index2 : speech_length * (index2 + 1)]
                  split_speech = np.reshape(split_speech, (1, speech_length, 1))
                  enhancement_speech = model.predict([split_speech])
                  predict = np.reshape(enhancement_speech, (speech_length, 1))
                  restore.extend(predict)
            restore = np.array(restore)

            sf.write("./model_pred/" + "{:04d}".format(index1+1) + ".wav", restore, sampling_rate, subtype='FLOAT')


            # scipy.io.wavfile.write("./model_pred/" + "{:04d}".format(index1+1) + ".wav", rate = sampling_rate, data = restore)



if __name__ == "__main__":
      tf.random.set_seed(seed = 42)
      parser = argparse.ArgumentParser(description = 'SETTING OPTION')
      parser.add_argument("--model", type = str, default = "dcunet20",         help = "Input model type")
      parser.add_argument("--load", type = str, default = "./model_save/dcunet2040.h5", help = "Input save model file")
      parser.add_argument("--data",  type = str, default = "./datasets/fn/",    help = "Input load unseen speech")
      parser.add_argument("--save",  type = str, default = "./model_pred/",      help = "Input save predict speech")
      args = parser.parse_args()

      model_type     = args.model
      load_file_path = args.load
      test_data_path = args.data
      pred_data_path = args.save


      if model_type == "naive_dcunet16":
            model = Naive_DCUnet16().model()
      elif model_type == "naive_dcunet20":
            model = Naive_DCUnet20().model()
      elif model_type == "dcunet16":
            model = DCUnet16().model()
      elif model_type == "dcunet20":
            model = DCUnet20().model()

      model.load_weights(load_file_path)
      model.summary()

      'READ SPEECH FILE'
      noisy_file_list = get_file_list(file_path = test_data_path)

      'INFERENCE'
      inference(path_list = noisy_file_list, save_path = pred_data_path)
      print("__END__")

      # example call:
      # python3.8 model_test.py --model naive_dcunet16 --load ./model_save/20210530T173510/dcunet200.h5 --data ./datasets_full/fn/small_noisy_set/
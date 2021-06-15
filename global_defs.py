

class MetaSeg:
  
  METASEG_MODEL_NAMES    = [ "best-loss-037epoch.bin", "best-loss-035epoch.bin" ]
  METASEG_CLASS_DTYPES   = [ "one_hot_classes", "probs" ]

  METASEG_MODEL_NAME     = METASEG_MODEL_NAMES[0]

  METASEG_READ_DATA_PATH = "/raid/zhuxihuan/data/body/expe6/train_data/"
  METASEG_MY_IO_PATH     = "/home/zhuxihuan/code/MetaSeg/expe6/pixel_train_measure/"

  #整张图的Marked，用于计算像素级块时的参考component
  METASEG_MY_PIXEL_IO_PATH  = "/home/zhuxihuan/code/MetaSeg/expe6/pixel_train_measure/"

  METASEG_READ_VAL_DATA_PATH = "/raid/zhuxihuan/data/body/expe6/val_data/"
  METASEG_MY_VAL_IO_PATH     = "/home/zhuxihuan/code/MetaSeg/expe6/pixel_val_measure/"

  METASEG_MY_VAL_PIXEL_IO_PATH = "/home/zhuxihuan/code/MetaSeg/expe6/pixel_val_measure/"
  
  METASEG = { "MODEL_NAMES"          : METASEG_MODEL_NAMES, \
              "CLASS_DTYPES"         : METASEG_CLASS_DTYPES, \
              "MODEL_NAME"           : METASEG_MODEL_NAME, \
              "PROBS_DIR"            : METASEG_READ_DATA_PATH + "probs/",\
              "GT_DIR"               : METASEG_READ_DATA_PATH + "GT/", \
              "IMG_DIR"              : METASEG_READ_DATA_PATH + "IMG", \
              "METRICS_DIR"          : METASEG_MY_IO_PATH     + "metrics/"     + METASEG_MODEL_NAME + "/", \
              "PIXEL_METRICS_DIR"    : METASEG_MY_PIXEL_IO_PATH     + "metrics/"     + METASEG_MODEL_NAME + "/", \
              "COMPONENTS_DIR"       : METASEG_MY_IO_PATH     + "components/"  + METASEG_MODEL_NAME + "/", \
              "PIXEL_COMPONENTS_DIR" : METASEG_MY_PIXEL_IO_PATH     + "components/"  + METASEG_MODEL_NAME + "/", \
              "VAL_METRICS_DIR"      : METASEG_MY_VAL_IO_PATH     + "metrics/"     + METASEG_MODEL_NAME + "/", \
              "VAL_PIXEL_METRICS_DIR": METASEG_MY_VAL_PIXEL_IO_PATH     + "metrics/"     + METASEG_MODEL_NAME + "/", \
              "VAL_COMPONENTS_DIR"   : METASEG_MY_VAL_IO_PATH     + "components/"  + METASEG_MODEL_NAME + "/", \
              "VAL_PIXEL_COMPONENTS_DIR"   : METASEG_MY_VAL_PIXEL_IO_PATH     + "components/"  + METASEG_MODEL_NAME + "/", \
              "RESULTS_DIR"          : METASEG_MY_IO_PATH     + "results/"     + METASEG_MODEL_NAME + "/", \
              "IOU_SEG_VIS_DIR"      : METASEG_MY_IO_PATH     + "iou_seg_vis/" + METASEG_MODEL_NAME + "/", \
              "STATS_DIR"            : METASEG_MY_IO_PATH     + "stats/"       + METASEG_MODEL_NAME + "/", \
              #expe1:1897
              #expe2:1675
              #expe3:3572
              #expe5:train 4393 val 2530
              #expe6:train 1800 val 194
              "NUM_IMAGES"           : 1800, \
              "NUM_CORES"            : 10, \
              "NUM_LASSO_AVERAGES"   : 10, \
              "NUM_LASSO_LAMBDAS"    : 50, \
              "COMPUTE_METRICS"      : False, \
              "VISUALIZE_METRICS"    : False, \
              "ANALYZE_METRICS"      : True, \
              "CLASS_DTYPE"          : METASEG_CLASS_DTYPES[1]
            }
  

  
  def __init__(self):
    for m in self.METASEG:
      setattr(self, m, self.METASEG[m])
  
  
  
  def get( self, name ):
    try:
      return getattr(self, name)
    except:
      print("MetaSeg:",name,"not found.")
      return 0
  
  
  
  def set( self, name, value ):
    if getattr(self, name):
      setattr( self, name, value )
  
  

  def print_attr( self ):
    for m in sorted(self.METASEG):
      print(m, ":", getattr(self, m) )


  
  def set_from_argv( self, argv ):
    cline = str()
    for i in range(len(argv)):
      cline += str(argv[i])+" "
    
    if "--" in cline:
      commands = cline.split("--")
      
      for c in commands[1:]:
        c0, c1 = c.split("=")[0], c.split("=")[1]
        
        if self.get( c0 ):
          dtype = type( self.get( c0 ) )
          self.set( c0, dtype(c1) )
    
    
    
    

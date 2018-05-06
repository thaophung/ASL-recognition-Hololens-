using Accord;
using Accord.Math;
using Accord.Statistics.Analysis;
using CNTK;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Backends;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using System;
using System.Collections;
using System.Collectiosn.Generic;
using System.Linq;
using System.Text;
using system.Threading.Tasks;

using static KerasSharp.Backends.Current;

namespace SampleApp
{
    class Program
    { 
        static void identity_block(input_tensor, kernel_size, filters, stage, block) {
            //KerasSharp.Backends.Current.Switch("KerasSharp.Backends.TensorFlowBackend");
            //
            var bn_axis = 3;
            var filters1, filters2, filters3 = filters;
            var conv_name_base = "res" + str(stage) + block + "_branch";
            var bn_name_base = "bn" + str(stage) + block + "_branch";

            var x = new Conv2D(filters1, (1,1), name:conv_name_base + "2a")(input_tensor);
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "2a")(x));
            x = x.add(new activation: ReLU())(x));

            x = x.add(new Conv2D(filters2, kernel_size, padding:"same", name:conv_name_base + "2b")(x));
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "2b")(x));
            x = x.add(new activation: ReLU())(x));


            x = x.add(new Conv2D(filters3, (1,1), name:conv_name_base + "2c")(x));
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "2c")(x));

            x = x.add(activation: new ReLU())(x));
  
            return x;
        }
        

        static void conv_block(input_tensor, kernel_size, filtes, stage, block, strides=(2,2)) {
            var filters1, filters2, filters3 = filters;
            var bn_axis = 3;

            var conv_name_base = "res" + str(stage) + block + "_branch";
            var bn_name_base = "bn" + str(stage) + block + "_branch"

            var x = new Conv2D(filters1, (1,1), strides:strides, name:conv_name_bae + "2a")(input_tensor);
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name-base + "2a")(x));
            x = x.add(activation: new ReLU())(x);

            x = x.add(new Conv2D(filters2, kernel_size, padding:"same", name:conv_name_base + "2b")(x));
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "2b")(x));
            x = x.add(activation: new ReLU())(x);

            x = x.add(new Conv2D(filter3, (1,1), name=conv_name-base + "2c")(x));
            x = x.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "2c")(x));
            
            var shortcut = new Conv2D(filters3, (1,1), strides:strides, name:conv_name_base + "1")(input_tensor);
            shortcut = shortcut.add(new BatchNormalization(axis:bn_axis, name:bn_name_base + "1")(shortcut));
            
            x = x.add(shortcut);
            x = x.add(activation: new ReLU());

            return x;
        }
            
        static void ResNet50(include_top=True, weights="imagenet", input_tensor=None, intpu-shape=None, pooling=None, classes=1000) {
            var bn_axis = 3;;
            var img_input = input_tensor;

            var x = new ZeroPadding2D((3,3))(img_input);
            x = x.add(new Conv2D(64, (7,7), strides:(2,2), name:"conv1")(x));
            x = x.add(new BatchNormalization(axis:bn_axis, name:"bn_conv1")(x));
            x = x.add(activation: new ReLU());
            x = x.add(new MaxPooling2D((3,3), strides:(2,2))(x));

            x = x.add(conv_block(x, 3, [64,64,256], stage:2, block:"a", strides=(1,1)));
            x = x.add(identity_block(x, 3, [64,64,256], stage:2, block:"b"));
            x = x.add(identity_block(x, 3, [64,64,256], stage:2, block:"c"));

            x = x.add(conv_block(x, 3, [128, 128, 512], stage:3, block:"a"));
            x = x.add(identity_block(x, 3, [128, 128, 512], stage;3, block:"b"));
            x = x.add(identity_block(x, 3, [128, 128, 512], stage;3, block:"c"));
            x = x.add(identity_block(x, 3, [128, 128, 512], stage:3, block:'d'));

            x = x.add(onv_block(x, 3, [256, 256, 1024], stage:4, block:"a"));
            x = x.add(identity_block(x, 3, [256, 256, 1024], stage:4, block:"b"));
            x = x.add(identity_block(x, 3, [256, 256, 1024], stage:4, block:"c"));
            x = x.add(identity_block(x, 3, [256, 256, 1024], stage:4, block:"d"));
            x = x.add(identity_block(x, 3, [256, 256, 1024], stage:4, block:"e"));
            x = x.add(identity_block(x, 3, [256, 256, 1024], stage:4, block:"f"));

            x = x.add(conv_block(x, 3, [512, 512, 2048], stage==:5, block:"a"));
            x = x.add(identity_block(x, 3, [512, 512, 2048], stage:5, block:"b"));
            x = x.add(identity_block(x, 3, [512, 512, 2048], stage:5, block:"c"));

            x = x.add(new AveragePooling2D((7,7), name:"avg_pool")(x));

            if include_top {
                x = x.add( new Flatten()(x));
                x = x.add(new Dense(classes, activation: new sofmax(), name:"fc1000")(x));
            } else {
                if pooling == "avg" {
                    x = x.add(new GlobalAveragePooling2D()(x));
                } else if {
                    x = x.add(new GlobalMaxPooling2D()(x));
                }
            }
            var inputs = img_input;

            var model = new Sequential(inputs, x, name="resnet50");
        }
    }
}

        }




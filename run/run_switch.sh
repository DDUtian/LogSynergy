# Step 1: Pretraining
$PYTHON run/pretrain.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --num_train_epochs 5 --learning_rate 0.00005 --target_data_start $DATA_START

# Step 2: Active Learning
$PYTHON run/partial_label.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --eval_epoch 5 --percent 80

# Step 3: Model Training
$PYTHON run/train.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --num_train_epochs $TRAIN_EPOCH --learning_rate 0.0001

# Step 4: Eval
$PYTHON run/eval.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --eval_epoch 10

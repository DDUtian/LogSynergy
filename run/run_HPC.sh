$PYTHON run/train.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --num_train_epochs $TRAIN_EPOCH --learning_rate 0.0001

$PYTHON run/eval.py $SOURCE_DOMAIN $TARGET_DOMAIN $DATA_PATH --batch_size $BATCH_SIZE --eval_epoch 10

from __future__ import print_function

import tensorflow as tf
import os.path
import shutil
import json


def train_model(inputStr):
    EXPORT_DIR = './output'

    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 1
    display_step = 1
    testDataSize = 10

    print("Str size: ", len(inputStr))

    data = json.loads(inputStr)
    # pprint(data)

    print("Data size: ", len(data))
    examplesSize = len(data) - testDataSize

    labels = [None] * examplesSize
    examples = [None] * examplesSize

    testLabels = [None] * testDataSize
    testExamples = [None] * testDataSize

    valuesIterator = iter(data.values());
    for i in range(examplesSize):
        item = next(valuesIterator)
        example = []
        for int_item in item['data']:
            example.append(float(int_item))
        examples[i] = example

        value = []
        for int_item in item['value']:
            value.append(float(int_item))
        labels[i] = value

    for i in range(testDataSize):
        item = next(valuesIterator)
        example = []
        for int_item in item['data']:
            example.append(float(int_item))
        testExamples[i] = example

        value = []
        for int_item in item['value']:
            value.append(float(int_item))
        testLabels[i] = value

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 625])  # mnist data image of shape 25*25=625
    y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

    # Set model weights
    W = tf.Variable(tf.zeros([625, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(examples) / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                batch_xs = examples[i:i + 1]
                batch_ys = labels[i:i + 1]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: testExamples, y: testLabels}))

        W_final = W.eval(sess)
        b_final = b.eval(sess)

    # Create new graph for exporting
    g = tf.Graph()
    with g.as_default():
        x_input = tf.placeholder(tf.float32, shape=[625], name="input")
        W_final = tf.constant(W_final, name="W_final")
        b_final = tf.constant(b_final, name="b_final")

        answer = tf.nn.softmax(tf.matmul(tf.reshape(x_input, [1, 625]), W_final) + b_final)
        OUTPUT = tf.reshape(answer, [-1], name="output")  # Softmax

        # x = tf.placeholder(tf.float32, [None, 625])
        # pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

        sess = tf.Session()
        # init = tf.global_variables_initializer()
        # sess.run(init)

        result = sess.run(OUTPUT, feed_dict={x_input: testExamples[0]})
        print("Result", result)
        print("Result", sess.run(tf.argmax(result)))

        graph_def = g.as_graph_def()
        tf.train.write_graph(graph_def, EXPORT_DIR, 'model_graph.pb', as_text=False)

    return True

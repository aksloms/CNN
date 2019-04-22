import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Pobranie danych
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Rozpoczecie sesji
sess = tf.InteractiveSession()

#Utworzenie placeholderow na dane
x = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

#Warstwa wejsciowa - 28x28 pixeli na obraz monochromatyczny (1)
# -1 - wartosc obliczana automatycznie reprezentujaca liczbe obrazow
input_layer = tf.reshape(x, [-1,28,28,1])

#Funkcje pomocnicze:
#Tworzenie zmiennej wagi
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#Tworzenie zmiennej bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Tworzenie zmiennej warstwy konwolucyjnej
def conv2d(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

#Tworzenie zmiennej warstwy pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#Utworzenie wag i biasu pierwszej warstwy konwolucyjnej
conv1_weight = weight_variable([5, 5, 1, 32])
conv1_bias = bias_variable([32])

#Polaczenie warstwy wejsciowej z pierwsza warstwa konwolucyjna (funkcja aktywujaca RELU)
conv1 = tf.nn.relu(conv2d(input_layer, conv1_weight) + conv1_bias)

#Utworzenie pierwszej warstwy pooling
pool1 = max_pool_2x2(conv1)

#Utworzenie wag i biasu drugiej warstwy konwolucyjnej
conv2_weight = weight_variable([5, 5, 32, 64])
conv2_bias = bias_variable([64])

#Polaczenie pierwszej warstwy pooling z druga warstwa konwolucyjna (funkcja aktywujaca RELU)
conv2 = tf.nn.relu(conv2d(pool1, conv2_weight) + conv2_bias)

#Utworzenie drugiej warstwy pooling
pool2 = max_pool_2x2(conv2)

#Przeksztalcenie drugiej warstwy pooling tak, aby mozna bylo polaczyc ja z warstwa gesta (dense)
pool2_to_dense = tf.reshape(pool2, [-1 , 7 * 7 * 64])

#Utworzenie wag i biasu warstwy gestej
dense_weight = weight_variable([7 * 7 * 64 , 1024])
dense_bias = bias_variable([1024])

#Utworzenie warstwy gestej
dense = tf.nn.relu(tf.matmul(pool2_to_dense, dense_weight) + dense_bias)

#Zabezpieczenie przed overfittingiem - keep_prob dla treningu - 0.5, dla ewaluacji 1
keep_prob = tf.placeholder(tf.float32) 
dropout = tf.nn.dropout(dense, keep_prob)

#Wagi i bias warstwy wyjsciowej
output_weight = weight_variable([1024, 10])
output_bias = bias_variable([10])

#Zdefiniowanie modelu
logits = tf.matmul(dropout, output_weight) + output_bias

#Obliczenie entropii krzyzowej (strat)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

#Zmniejszenie entropii krzyzowej (optymalizacja)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Znalezienie prawidlowego przewidywania i jego prawdopodobieństwa poprawnosci
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Inicjalizacja zmiennych dla sesji
sess.run(tf.global_variables_initializer())

#Import czasu - kontrola predkosci
import time

#Definicja liczby krokow
num_steps = 3000

#Definicja czestosci wyswietlania postepow (tu co 100 krokow)
display_every = 100

start_time = time.time()
end_time = time.time()

#Petla po kolejnych krokach
for i in range(num_steps):
    #trening
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})

    #ewaluacja co 100 krokow
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], labels: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("Krok {0}, uplynęlo {1:.2f} sekund, trafnosc = {2:.3f}%".format(i, end_time-start_time, train_accuracy * 100.0))


end_time = time.time()

#Wyswietlenie podsumowania
print("Calkowity czas treningu wyniosl {0:.2f} sekund".format(end_time-start_time))

print("Skutecznosc sieci dla danych testowych {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})*100.0))

#Zamkniecie sesji
sess.close()


























































































































































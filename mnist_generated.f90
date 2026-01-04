program mnist_network

use nf, only: conv, dense, dropout, flatten, input, label_digits, locallyconnected, maxpool, network, relu, reshape, sgd, sigmoid, softmax, tanh

implicit none

type(network) :: net

real, allocatable :: training_images(:,:), training_labels(:)
real, allocatable :: validation_images(:,:), validation_labels(:)
real, allocatable :: testing_images(:,:), testing_labels(:)
integer :: n
integer, parameter :: num_epochs = 10

call load_mnist(training_images, training_labels, &
                validation_images, validation_labels, &
                testing_images, testing_labels)

! Construct network from configuration
net = network([ &
        input(784), &
        reshape([1, 28, 28]), &
        conv(8, 3, 3, relu), &
        maxpool(2, 2, 2), &
        conv(16, 3, 3, relu), &
        maxpool(2, 2, 2), &
        dense(10, softmax) &
    ])

call net%print_info()

epochs: do n = 1, num_epochs

    call net%train( &
        training_images, &
        label_digits(training_labels), &
        batch_size=10, &
        epochs=1, &
        optimizer=adam(learning_rate=0.001) &
    )

    print '(a,i2,a,f5.2,a)', 'Epoch ', n, ' done, Accuracy: ', accuracy( &
        net, validation_images, label_digits(validation_labels)) * 100, ' %'

end do epochs

print '(a,f5.2,a)', 'Testing accuracy: ', &
    accuracy(net, testing_images, label_digits(testing_labels)) * 100, '%'

contains

real function accuracy(net, x, y)
    type(network), intent(in out) :: net
    real, intent(in) :: x(:,:), y(:,:)
    integer :: i, good
    good = 0
    do i = 1, size(x, dim=2)
        if (all(maxloc(net%predict(x(:,i))) == maxloc(y(:,i)))) then
            good = good + 1
        end if
    end do
    accuracy = real(good) / size(x, dim=2)
end function accuracy

end program mnist_network

class InvalidKernel(Exception):
    def __init__(self, kernel):
        message = "Kernel \"{}\" not recognized".format(kernel)
        super(InvalidKernel, self).__init__(message)

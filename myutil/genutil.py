import os


def create_directory(directory_name):
    """Create directory safely after checking if it already 
    exists or not.
    
    Arguments:
        directory_name {str} -- name of the directory 
    """
    if os.path.exists(directory_name):
        print("Directory: {}/ already exists.".format(directory_name))
    else:
        os.makedirs(directory_name)
        print("Directory: {}/ created.".format(directory_name))


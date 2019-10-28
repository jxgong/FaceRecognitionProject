##FaceLocker V0.1

Facelocker is a software that scrambles text files to the point of unreadability. Instead of using a password, it uses the client's face to lock and unlock files.

###To install:

You're going to need OpenCV and Numpy to run this. To get these modules, go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv and http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy to download the packages that correspond to your system and python version.

After that, use the command line to go to the directory you downloaded using cd [path].

Finally, use the command "pip install [file name here]" to install the files.

This project also makes use of Haar cascades, which are included in the project library.

All you need to do is change the path in the detect_face function to the path for the Haar cascades.

Make sure that if you move the file anywhere, lockIcon.png is moved along with it.

Before you start: It's a good idea to change the initial path in init() to your documents folder, so that you don't have to wade through miscellaneous program files to get to the file you want.

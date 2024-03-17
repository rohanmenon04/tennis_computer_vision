Tennis computer vision model, which takes in a video of a singles point and tracks the players, ball and keypoints on the court. Measures the speed of the players and the ball and displays the path of the ball in real time. 

The video needs to be from a fixed camera, and currently only accurately with Djokovic and Sonego. In order to correctly adjust the model to be able to accurately compute and display the mini-court for any two players the player heights will need to be adjusted to match the heights of the players in the video. This can be updated in the constants folder and in the *__init__*.py file.

Output video example can be seen here: https://drive.google.com/file/d/1_xtSw-yVIumiWeDu0Yfo2_sgqBFnDdnx/view?usp=sharing

Project idea credits: https://www.youtube.com/watch?v=L23oIHZE14w&t

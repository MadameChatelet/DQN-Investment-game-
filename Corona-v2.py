import random

import numpy as np
from collections import deque 
#special kind of list for the memory of our agent
#deque is like a list where you can add things from the top or end of the list
import tensorflow as tf 
from tensorflow.keras.models import Sequential # BUILD OUR NURAL NETWORK
#to aproximate optimal Q
from tensorflow.keras.layers import Dense#we will just use dense layers
from tensorflow.keras.optimizers import Adam#as optimizer we will use Adam, estoCASTIC GRADIENT DESCENT
import os #to create directories 
import turtle
# 1. define environment

#we have 4 different state
#and 4 different actions

state_size = 4
action_size = 4
batch_size = 32 #for our gradient descent, hyperparameter you can vary for powers of two

n_episodes = 1001#number of games we want our agent to play
#number of games we play, provides as with more data for training
#in each of this episodes, we are gonna randomly remember,
#some of the things that happen in that episode
#we are gonna use that memory to train de RL agent


os.environ['CUDA_DEVICE_ORDER'] = "PCI_SUS_ID"#to run it in our cpu
os.environ['CUDA_VISIBLE_DEVICES'] = ""
output_dir = 'model_output/cartpole'# define a directory, to store model output

#this code if to not allow to create the directory if alreadt exists

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class Corona():
    
    def __init__(self):
        


        self.done = False
        self.reward = 0
        self.people, self.spending = 0, 6000
        

        #Background
        self.win = turtle.Screen() #create screen
        self.win.title('After Corona World') 
        self.win.bgcolor('green')
        self.win.tracer(0)
        self.win.setup(width = 600, height = 600)

        #house
        self.house = turtle.Turtle()
        self.house.shape('square')
        self.house.speed(0)
        self.house.shapesize(stretch_wid=3, stretch_len=3)   # Streach the length of square by 5 
        self.house.penup()
        self.house.color('white')       # Set the color to white
        self.house.goto(0, 0) 
        self.roof = turtle.Turtle()
        self.roof.shape('triangle')
        self.roof.tilt(-30)
        self.roof.speed(0)
        self.roof.shapesize(stretch_wid=3, stretch_len=3)   # Streach the length of square by 5 
        self.roof.penup()
        self.roof.color('red')       # Set the color to white
        self.roof.goto(0, 50-3) 

        #super1
        self.super1 = turtle.Turtle()
        self.super1.shape('square')
        self.super1.speed(0)
        self.super1.shapesize(stretch_wid=2.5, stretch_len=4)   # Streach the length of square by 5 
        self.super1.penup()
        self.super1.color('brown')       # Set the color to white
        self.super1.goto( 160 , 0) 


        #super2
        self.super2 = turtle.Turtle()
        self.super2.shape('square')
        self.super2.speed(0)
        self.super2.shapesize(stretch_wid=2.5, stretch_len=4)   # Streach the length of square by 5 
        self.super2.penup()
        self.super2.color('yellow')       # Set the color to white
        self.super2.goto( -160, 0 ) 

        #super3
        self.super3 = turtle.Turtle()
        self.super3.shape('square')
        self.super3.speed(0)
        self.super3.shapesize(stretch_wid=2.5, stretch_len=4)   # Streach the length of square by 5 
        self.super3.penup()
        self.super3.color('orange')       # Set the color to white
        self.super3.goto( 0, -160) 

        #cinema
        self.park = turtle.Turtle()
        self.park.shape('circle')
        self.park.speed(0)
        self.park.shapesize(stretch_wid=3.5, stretch_len=6)   # Streach the length of square by 5 
        self.park.penup()
        self.park.color('blue')       # Set the color to white
        self.park.goto( 0, 160)
        
      
        
        
        # Bilbo Beggins
        self.bilbo = turtle.Turtle()      # Create a turtle object
        self.bilbo.speed(0)
        self.bilbo.shape('circle')        # Select a circle shape
        self.bilbo.color('violet')           # Set the color to red
        self.bilbo.penup()
        self.bilbo.goto(0, 0)           # Place the shape in middle

        #names
        self.name = turtle.Turtle()
        self.name.speed(0)
        self.name.color('black')
        self.name.penup()
        self.name.hideturtle()
        
        #CINEMA
        self.name.goto(110,150)
        self.name.write("P: 200\n$: 8", align='center', font=('Courier', 14, 'normal'))
        self.name.goto(0,200)
        self.name.write("CINEMA", align='center', font=('Courier', 14, 'normal'))
        
        #PIZZERIA
        self.name.goto(160+70,-10)
        self.name.write("P: 30\n$: 20", align='center', font=('Courier', 14, 'normal'))
        self.name.goto(160, 30)
        self.name.write("PIZZERIA", align='center', font=('Courier', 14, 'normal'))        

        #COFFE SHOP
        self.name.goto(-160-70,-10)
        self.name.write("P: 6\n$: 2", align='center', font=('Courier', 14, 'normal'))
        self.name.goto(-160, 30)
        self.name.write("COFFE SHOP", align='center', font=('Courier', 14, 'normal')) 
        
        #SUPERMARKET
        self.name.goto(70,-170)
        self.name.write("P: 50\n$: 100", align='center', font=('Courier', 14, 'normal'))
        self.name.goto(0,-160+30)
        self.name.write("SUPERMARKET", align='center', font=('Courier', 14, 'normal')) 
        
        #Scorecard
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0,250)
        self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))

        
        # Keyboard Control
        self.win.listen()
        self.win.onkey(self.bilbo_right, 'Right')   # call bilbo_right on right arrow key
        self.win.onkey(self.bilbo_left, 'Left')     # call bilbo_left on right arrow key
        self.win.onkey(self.bilbo_up, 'Up')   # call bilbo_up on up arrow key
        self.win.onkey(self.bilbo_down, 'Down')     # call bilbo_down on down arrow key

    #let's add Bilbo left, right, up, down movement

    def bilbo_right(self):
        x = self.bilbo.xcor()        # Get the x position of bilbo
        if -20 < x < 20 :
            self.bilbo.goto(160,0)    # increment the x position by 160
        if  -20 > x  :
            self.bilbo.goto(0,0)


    def bilbo_left(self):
        x = self.bilbo.xcor()        # Get the x position of bilbo
        if x > 20 :
            self.bilbo.goto(0,0)    # decrement the x position by 160
        if  -20 < x < 20  :
            self.bilbo.goto(-160,0)

    def bilbo_up(self):

        y = self.bilbo.ycor()        # Get the y position of bilbo
        if -20 < y < 20 :
            self.bilbo.goto(0,160)    # decrement the x position by 160

        if   y < -20  :
            self.bilbo.goto(0,0)    # increment the y position by 160

    def bilbo_down(self):
        y = self.bilbo.ycor()        # Get the y position of bilbo
        if y > 20 :
            self.bilbo.goto(0,0)    # decrement the x position by 160
        if  -20 < y < 20 :
            self.bilbo.goto(0,-160)    # decrement the y position by 160

# -------------RL----------
    #ACTIONS              | REWARDS
    #  0 move left    ->    (-0.1 Bilbo moves)
    #  1 move right   ->    (-0.1 Bilbo moves)
    #  2 move up      ->    (-0.1 Bilbo moves)
    #  3 move down    ->    (-0.1 Bilbo moves)
    
    def reset(self):
        
        self.bilbo.goto(0,0)
        return [self.bilbo.xcor()*0.01, self.bilbo.ycor()*0.01, self.spending, self.people]
    
    def step(self, action):
        
        self.reward = 0
        self.done = 0
        
        if action == 0:
            self.bilbo_left()
            self.reward -= .1
            
        if action == 1:
            self.bilbo_right()
            self.reward -= .1
            
        if action == 2:
            self.bilbo_up()
            self.reward -= .1
            
        if action == 3:
            self.bilbo_down()
            self.reward -= .1  
            
        self.run_frame()
        
        state = [self.bilbo.xcor()*0.01, self.bilbo.ycor()*0.01, self.spending, self.people]#, self.people, self.spending]
        
        #maybe put people and spendings for each supermarket?
        
        return self.reward, state, self.done
    
    
    def run_frame(self):
        
        
        self.win.update()
        
        if self.spending > 4000 :# positive rewards --feelings
        
            if self.bilbo.pos() == (0, 160):#if Bilbo goes to the cinema
                self.bilbo.sety(160-1)
                self.reward += 3
                self.people += 200
                self.spending -= 8
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))



            if self.bilbo.pos() == (160, 0):#if Bilbo goes to the Supermarket 1
                self.bilbo.setx(160-1)
                self.reward += 3 #because he buys grosseies
                self.people += 30
                self.spending -= 20
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (0, -160):#if Bilbo goes to the Supermarket 2
                self.bilbo.sety(-160+1)
                self.reward -= 3 #because he buys grosseies
                self.people += 50
                self.spending -= 100
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (-160, 0):#if Bilbo goes to the Supermarket 3
                self.bilbo.setx(-160+1)
                self.reward -= 3 #because he buys grosseies
                self.people += 6
                self.spending -= 5
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (0, 0):#if Bilbo goes home
                self.bilbo.setx(+1)
                self.reward += 4 #because people is 0
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


        
        if self.spending <= 4000 :# now everything is negative, we want to converge to the solution fast
            
            if self.bilbo.pos() == (0, 160):#if Bilbo goes to the cinema
                self.bilbo.sety(160-1)
                self.reward -= 3
                self.people += 200
                self.spending -= 8
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))



            if self.bilbo.pos() == (160, 0):#if Bilbo goes to the pizzeria
                self.bilbo.setx(160-1)
                self.reward -= 3 #because he buys grosseies
                self.people += 30
                self.spending -= 20
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (0, -160):#if Bilbo goes to the coffe shop
                self.bilbo.sety(-160+1)
                self.reward += 3 #because he buys grosseies
                self.people += 50
                self.spending -= 100
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (-160, 0):#if Bilbo goes to the Supermarket 3
                self.bilbo.setx(-160+1)
                self.reward += 3 #because he buys grosseies
                self.people += 6
                self.spending -= 5
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))


            if self.bilbo.pos() == (0, 0):#if Bilbo goes home
                self.bilbo.setx(+1)
                self.reward += 4 #because he stays at home
                self.score.clear()
                self.score.write("People: {}  Savings: {}".format(self.people, self.spending), align='center', font=('Courier', 24, 'normal'))

  
            
        if self.spending <= 500 or self.people > 10000 :#or self.people >= 4000
            self.done = True
            self.spending = 6000 #savings
            self.people = 0


class DQNAgent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0]) # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state)[0])) # (maximum target Q based on future action a')
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_dqn(n_episodes, state_size, action_size):

    loss = []
    
    agent = DQNAgent(state_size, action_size) #initialize our agent
    
    env = Corona()
    done = False # episode has not ended
    for e in range(n_episodes):#each episode in the range we said (2000)

        state = env.reset()#start eah episode at the begining f the episode
        state = state
        state = np.reshape(state, [1, state_size])#reshape this states, and transpode them so they can fit align with the neural network

        for time in range(5000):#set max number of time-steps the eisode can run for,
        #max game-time is 5000 time steps

            #env.render() # for now doesn't work
            action = agent.act(state) #pass current state, so that it can take some initial action


            reward, next_state, done = env.step(action)
            #after our agent has taken an action
            #we can use that action to pass to the environment 
            #and get our NExT_STATE and our NEXT_REWARD from the environment
            #we also get a boolean -> if the game is done or not
            #we take that by taking an step forward

            reward = reward if not done else -0.1#calculate our reward
            #if you are not done, we get a penalty.
            #penalize poor actions

            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            #to remember the previous time-step..., all stuff we want to remember

            state = next_state# what state was in the previous iteration

            if done: #if episode ended, lets print how the agent performed
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e, n_episodes, time, agent.epsilon))
                #epsilon, good place to look if the agent is performing well
                break

            #train our data
            #give time to the agent to update his weights, so he can improves for future iterations
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e% 50 == 0:
            agent.save(output_dir + 'weights' + '{:04d}'.format(e) + '.hdf5')#we specify format to print
            
    
        loss.append(reward)
    return loss


if __name__ == '__main__':

    ep = 50
    loss = train_dqn(ep,  state_size, action_size)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
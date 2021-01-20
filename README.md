# Neural Network

To successfully compile and run the program. External libraries including pandas, matplotlib, sklearn, seaborn need to be installed.
Click View -> Tool Window -> Terminal. 
Type “ pip install pandas”.
“pip install matplotlib”
“pip install sklearn”.
“pip install seaborn”.
When the code is executed. User is asked to choose the activation function. Enter “1” to choose Sigmoid, “2” for Tanh and “3” for Relu.

Statlog (Heart) Data Set from UCI Machine Learning Repository was chosen to train and test the Neural Network model. 
This dataset contains 13 attributes
Attribute Information:
------------------------------------------------------------------------------------------------------------------------
      -- 1. age       
      -- 2. sex       
      -- 3. chest pain type (4 values)      
      -- 4. resting blood pressure  
      -- 5. serum cholestoral in mg/dl      
      -- 6. fasting blood sugar > 120 mg/dl       
      -- 7. resting electrocardiographic results (values 0,1,2) 
      -- 8. maximum heart rate achieved  
      -- 9. exercise induced angina    
      -- 10. oldpeak = ST depression induced by exercise relative to rest   
      -- 11. the slope of the peak exercise ST segment     
      -- 12. number of major vessels (0-3) colored by flourosopy        
      -- 13.  thal: 3 = normal; 6 = fixed defect; 7 = reversable defect     
Attributes types
-----------------
Real: 1,4,5,8,10,12 <br />
Ordered:11,<br />
Binary: 2,6,9<br />
Nominal:7,3,13<br />
Variable to be predicted
------------------------------------------------------------------------------------------------------------------------
Absence (0) or presence (1) of heart disease
------------------------------------------------------------------------------------------------------------------------
Training and Testing Error Summary with specific number of Iterations and Learning Step

<img src="https://user-images.githubusercontent.com/54776410/105114827-4e91d200-5a8d-11eb-855c-29912903eeb8.png">

It is observed that with the same Iteration number and learning rate, ReLu gave the lowest training and testing error, the next one is Sigmoid and then Tanh. More details are showed in the output files.

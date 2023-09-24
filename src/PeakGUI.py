from PyQt5 import QtWidgets, uic
import sys
import joblib
import torch 

mapping_age = {"< 25" : 6, "25 - 34" : 0, "35 - 44" : 1, "45 - 54" : 2, "55 - 64" : 3, "65 - 74" : 4, "> 75" : 7}
mapping_sex = {"Male": 1, "Female": 2}
mapping_race = {"White": 5, "Asian" : 2, "Black or African American" : 3, "American Indian or Alaska Native" : 1, "Asian Indian" : 21, "Chinese" : 22, "Filipino" : 23, "Japanese" : 24, "Korean" : 25, "Vietnamese": 26, "Other Asian" : 27, "Native Hawaiian" : 41, "Native Hawaiian or Other Pacific Islander" : 4, "Samoan": 43, "Other Pacific Islander" : 44, "Guamanian or Chamarro": 42}
mapping_ethnicity = {"Hispanic or Latino" : 2, "Not Hispanic or Latino" : 4}
occupancy = {"Principal residence": 1, "Second residence" : 2, "Investment property" : 3}
dwelling = {"Single Family": 2 , "Multifamily": 1}
mapping_loan_type = {"Conventional": 1, "Federal Housing Administration insured" : 2, "Veteran Affairs guaranteed" : 3, "USDA Rural Housing Service or Farm Service Agency guaranteed" : 4}
mapping_loan_purpose = {"Home purchase" : 1, "Home improvement" : 2, "Refinancing" : 31, "Cash-out refinancing" : 32, "Other purpose" : 4}



def inference(input):
    #y_pred = loan_eligibility_model(torch.tensor(input).to(torch.float32))
    #return [torch.argmax(y_pred)]
    model = joblib.load('random_forest_model.pkl')
    predictions = model.predict(input)
    return predictions
    #print(predictions)


#1. Create the model class
class neural_network_model(torch.nn.Module):
    def __init__(self) -> None: #Constructor, but also we don't need *args and **kwargs
        super().__init__()
        self.input = torch.nn.Linear(13, 32)
        #self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(32, 32)
        self.activation2 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(32, 32)
        self.activation3 = torch.nn.LeakyReLU()
        self.linear3 = torch.nn.Linear(32, 16)
        self.activation4 = torch.nn.LeakyReLU()
        self.linear4 = torch.nn.Linear(16, 10)
        self.activation5 = torch.nn.LeakyReLU()

    def forward(self, x): #Note that x should be an input tensor
        x = self.input(x)
        #x = self.activation1(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        x = self.activation3(x)
        x = self.linear3(x)
        x = self.activation4(x)
        x = self.linear4(x)
        x = self.activation5(x)
        #print(x)
        return x
    
loan_eligibility_model = neural_network_model()
state_dict = torch.load("DNN Model")
loan_eligibility_model.load_state_dict(state_dict)


class UI(QtWidgets.QMainWindow):
   def __init__(self):
       super(UI, self).__init__()
       uic.loadUi('ui.ui',self)
       self.calc_button = self.findChild(QtWidgets.QPushButton, "calButton")
       self.calc_button.clicked.connect(self.calc_button_pressed)
       self.reset_button = self.findChild(QtWidgets.QPushButton, "resButton")
       self.age_input = self.findChild(QtWidgets.QComboBox, "ageBox")
       self.sex_input = self.findChild(QtWidgets.QComboBox, "sexBox")
       self.race_input = self.findChild(QtWidgets.QComboBox, "raceBox")
       self.ethn_input = self.findChild(QtWidgets.QComboBox, "ethBox")
       self.occ_input = self.findChild(QtWidgets.QComboBox,  "occBox")
       self.dwe_input = self.findChild(QtWidgets.QComboBox, "dweBox")
       self.inc_input = self.findChild(QtWidgets.QLineEdit, "incBox")
       self.debt_input = self.findChild(QtWidgets.QLineEdit, "debBox")
       self.pro_input = self.findChild(QtWidgets.QLineEdit, "proBox")
       self.amm_input = self.findChild(QtWidgets.QLineEdit, "ammBox")
       self.typ_input = self.findChild(QtWidgets.QComboBox, "typBox")
       self.ter_input = self.findChild(QtWidgets.QLineEdit, "terBox")
       self.pur_input = self.findChild(QtWidgets.QComboBox, "purBox")
       self.rat_input = self.findChild(QtWidgets.QLineEdit, "ratBox")
       self.rea_output = self.findChild(QtWidgets.QLabel, "reaLabel")
       self.app_output = self.findChild(QtWidgets.QLabel, "appLabel")
       self.den_output = self.findChild(QtWidgets.QLabel, "denBox")
       self.show()
   def calc_button_pressed(self):
       age = mapping_age[self.age_input.currentText()]
       sex = mapping_sex[self.sex_input.currentText()]
       race = mapping_race[self.race_input.currentText()]
       ethn = mapping_ethnicity[self.ethn_input.currentText()]
       occ = occupancy[self.occ_input.currentText()]
       dwe = dwelling[self.dwe_input.currentText()]
       inc = int(self.inc_input.text())
       debt = int(self.debt_input.text())
       pro = int(self.pro_input.text())
       amm = int(self.amm_input.text())
       typ = mapping_loan_type[self.typ_input.currentText()]
       ter = int(self.ter_input.text())
       pur = mapping_loan_purpose[self.pur_input.currentText()]
       rat = int(self.rat_input.text())
       lst = [[dwe, ethn, race, sex, typ, pur, amm, rat, ter, pro, occ, age, inc]]
       if return_loan.loan_status(inference(lst)) == "Approved":
           self.app_output.setText("Approved")
           self.app_output.setStyleSheet("color: green;")
       else:
           self.app_output.setText("Denied")
           self.app_output.setStyleSheet("color: red;")
           self.den_output.setText("{}".format(inference(lst)[0].item()))
           self.den_output.setStyleSheet("color: yellow;")


class return_loan:
    def loan_status(re_lst):
        if re_lst[0] == 10:
            return ("Approved")
        return ("Denied")


app = QtWidgets.QApplication(sys.argv)
window = UI()
app.exec_()
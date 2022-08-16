from flask import Flask, render_template, request
import pandas as pd
import random

app = Flask(__name__)

products = ['Baby Food', 'Diapers', 'Formula', 'Lotion', 'Baby wash', 'Wipes', 'Fresh Fruits', 'Fresh Vegetables', 'Beer', 'Wine', 'Club Soda', 'Sports Drink', 'Chips', 'Popcorn', 'Oatmeal', 'Medicines', 'Canned Foods', 'Cigarettes', 'Cheese', 'Cleaning Products', 'Condiments', 'Frozen Foods', 'Kitchen Items', 'Meat', 'Office Supplies', 'Personal Care', 'Pet Supplies', 'Sea Food','Spices']

import h2o
h2o.init()

Model_Baby_Food = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_1_20211020_165337')
Model_Diapers = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_2_20211020_165531')
Model_Formula = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_3_20211020_165703')
Model_Lotion = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_4_20211020_165834')
Model_Baby_wash = h2o.load_model('./models/GBM_5_AutoML_5_20211020_165958')
Model_Wipes = h2o.load_model('./models/StackedEnsemble_AllModels_5_AutoML_6_20211020_170123')
Model_Fresh_Fruits = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_7_20211020_170250')
Model_Fresh_Vegetables = h2o.load_model('./models/GBM_grid_1_AutoML_8_20211020_170420_model_1')
Model_Beer = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_9_20211020_170548')
Model_Wine = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_10_20211020_170712')
Model_Club_Soda = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_11_20211020_170907')
Model_Sports_Drink = h2o.load_model('./models/GBM_1_AutoML_12_20211020_171048')
Model_Chips = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_13_20211020_171216')
Model_Popcorn = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_14_20211020_171355')
Model_Oatmeal = h2o.load_model('./models/StackedEnsemble_AllModels_3_AutoML_15_20211020_171534')
Model_Medicines = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_16_20211020_171702')
Model_Canned_Foods = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_17_20211020_171855')
Model_Cigarettes = h2o.load_model('./models/StackedEnsemble_AllModels_3_AutoML_18_20211020_172031')
Model_Cheese = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_19_20211020_172213')
Model_Cleaning_Products = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_20_20211020_172407')
Model_Condiments = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_21_20211020_172536')
Model_Frozen_Foods = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_22_20211020_172656')
Model_Kitchen_Items = h2o.load_model('./models/StackedEnsemble_BestOfFamily_2_AutoML_23_20211020_172854')
Model_Meat = h2o.load_model('./models/GBM_5_AutoML_24_20211020_173021')
Model_Office_Supplies = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_25_20211020_173148')
Model_Personal_Care = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_26_20211020_173318')
Model_Pet_Supplies = h2o.load_model('./models/StackedEnsemble_BestOfFamily_4_AutoML_27_20211020_173508')
Model_Sea_Food = h2o.load_model('./models/StackedEnsemble_AllModels_4_AutoML_28_20211020_173635')
Model_Spices = h2o.load_model('./models/GBM_1_AutoML_29_20211020_173827')

data = pd.read_pickle("./product_recommendation.pkl")

@app.route('/')
def index():
    items = set()
    x = True
    while x:
        s = products[random.randint(0,28)]
        s = s.replace(' ', '')
        s = s.lower()
        items.add(s)
        if len(items) == 4:
            x = False
    return render_template('index.html', items=list(items))

@app.route('/predictionCus', methods=['POST', 'GET'])
def predictionCus():
    items = set()
    x = True
    while x:
        s = products[random.randint(0,28)]
        s = s.replace(' ', '')
        s = s.lower()
        items.add(s)
        if len(items) == 8:
            x = False
        
    return render_template('predictionCus.html', items=list(items))



@app.route('/buyProducts', methods=['POST', 'GET'])
def buyProducts():
    items = []
    if request.method == 'POST':
        items = request.form.getlist('cb')
        prediction = set()
        for j in items:
            for i in data.index:
                try:
                    g = list(data['antecedents'].iloc[i])
                    if j in g:
                        m = list(data['consequents'].iloc[i])
                        for z in g:
                            prediction.add(z)
                            if len(prediction) == 8:
                                break
                except:
                    pass
        print(prediction)
        items = []
        for i in prediction:
            s = i.replace(' ', '')
            s = s.lower()
            items.append(s)
        print(items)
        x = True
        items2 = set()
        while x:
            s = products[random.randint(0,28)]
            s = s.replace(' ', '')
            s = s.lower()
            if s not in items:
                items2.add(s)
            if len(items2) == 8-len(items) or len(items) > 8-len(items):
                x = False
        items = items+list(items2)
        print(items)
        return render_template('prediction.html', items=items)

    
    return render_template('buyProducts.html')

@app.route('/prediction')
def prediction():
    items = set()
    x = True
    while x:
        y = random.randint(0,28)
        s = products[y]
        s = s.replace(' ', '')
        s = s.lower()
        items.add(s)
        if len(items) == 8 or len(items) > 8:
            x = False
    return render_template('prediction.html', items=list(items))

@app.route('/cusotmerinfo', methods=['POST', 'GET'])
def customerInfo():
    if request.method == 'POST':
        age = request.form['age']
        generation = request.form['generation']
        postalcode = request.form['postalcode']
        gendercode = request.form['gendercode']
        creditcardtype = request.form['creditcardtype']

        l = {'GenderCode':gendercode,'POSTAL_CODE':postalcode,'AGE':age,'CREDITCARD_TYPE':creditcardtype,'GENERATION':generation}
        print(l)
        pdata = h2o.H2OFrame(l)

        items = set()
        print(items)

        x = Model_Baby_Food.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Baby Food')
        print(items)

        x = Model_Diapers.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Diapers')
        print(items)

        x = Model_Formula.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Formula')
        print(items)

        x = Model_Lotion.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Lotion')
        print(items)

        x = Model_Baby_wash.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Baby wash')
        print(items)

        x = Model_Wipes.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Baby Wipes')
        print(items)

        x = Model_Fresh_Fruits.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Fresh Fruits')
        print(items)

        x = Model_Fresh_Vegetables.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Fresh Vegetables')
        print(items)

        x = Model_Beer.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Beer')
        print(items)

        x = Model_Wine.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Wine')
        print(items)

        x = Model_Club_Soda.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Club Soda')
        print(items)

        x = Model_Sports_Drink.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Sports Drink')
        print(items)

        x = Model_Chips.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Chips')
        print(items)

        x = Model_Popcorn.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Popcorn')
        print(items)

        x = Model_Oatmeal.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Oatmeal')
        print(items)

        x = Model_Medicines.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Medicines')
        print(items)

        x = Model_Canned_Foods.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Canned Foods')
        print(items)

        x = Model_Cigarettes.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Cigarettes')
        print(items)

        x = Model_Cheese.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Cheese')
        print(items)

        x = Model_Cleaning_Products.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Cleaning Products')
        print(items)

        x = Model_Condiments.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Condiments')
        print(items)

        x = Model_Frozen_Foods.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Frozen Foods')
        print(items)

        x = Model_Kitchen_Items.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Kitchen Items')
        print(items)

        x = Model_Meat.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Meat')
        print(items)

        x = Model_Office_Supplies.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Office Supplies')
        print(items)

        x = Model_Personal_Care.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Personal Care')
        print(items)

        x = Model_Pet_Supplies.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Pet Supplies')
        print(items)

        x = Model_Sea_Food.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Sea Food')
        print(items)

        x = Model_Spices.predict(pdata)
        y = x.as_data_frame()
        if y.iloc[0][0] == 1:
            items.add('Spices')
        print(items)

        prediction = set(items)
        items = list()
        for i in prediction:
            s = i.replace(' ', '')
            s = s.lower()
            items.append(s)
        print(items)

        x = True
        items2 = set()
        while x:
            s = products[random.randint(0,28)]
            s = s.replace(' ', '')
            s = s.lower()
            if len(items2) >= 8-len(items):
                x = False
            if s not in items:
                items2.add(s)
            
        print(items)
        print(items2)
        items = items+list(items2)
        print(items)

        return render_template('predictionCus.html',items=items)

    return render_template('customerInfo.html')



if __name__ == '__main__':
    app.run(debug=True)
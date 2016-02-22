# Airbnb Kaggle Competition: *New User Bookings*

This repository contains the code developed for the [Airbnb Kaggle
competition][competition]. It's written in **Python 2.7**. 

The code produces a prediction with a score around 0.88670, winner of the 3rd 
place out of 1463 teams in the competition.

The entire run can take several hours and you may run into memory
issues with less than 8GB RAM.

[competition]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

## Description (from competition [website][competition])

Where will a new guest book their first travel experience?

Instead of waking to overlooked "Do not disturb" signs, Airbnb travelers find themselves rising with the birds in a whimsical treehouse, having their morning coffee on the deck of a houseboat, or cooking a shared regional breakfast with their hosts.

New users on Airbnb can book a place to stay in 34,000+ cities across 190+ countries. By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

In this competition, the challenges is to predict in which country a new user will make his or her first booking. 
In this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics. You are asked to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: **US**, **FR**, **CA**, **GB**, **ES**, **IT**, **PT**, **NL**, **DE**, **AU**, **NDF** (no destination found), and **other**. Please note that **NDF** is different from **other** because **other** means there was a booking, but is to a country not included in the list, while **NDF** means there wasn't a booking.

The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010. 

## Data


Due to the [*Competition Rules*][rules], the data sets can not be shared. If
you want to take a look at the data, head over the [competition][competition]
page and download it.

[rules]: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/rules

## Main Ideas of this solution


## Requirements

To execute the code in this repository you will need the next Python packages:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [SciPy](http://www.scipy.org/)
- [SciKit-Learn](http://scikit-learn.org/stable/)
- [XGBoost](https://github.com/dmlc/xgboost)
- [Keras](https://github.com/fchollet/keras)
- [LetorMetric](https://gist.github.com/adamliesko/dddaa52c4b05b9a581b3)

## Resources


## License

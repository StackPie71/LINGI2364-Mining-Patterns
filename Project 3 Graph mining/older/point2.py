"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from sklearn import metrics
import operator
from sklearn.tree import DecisionTreeClassifier

from gspan_mining import gSpan
from gspan_mining import GraphDatabase


class PatternGraphs:
	"""
	This template class is used to define a task for the gSpan implementation.
	You should not modify this class but extend it to define new tasks
	"""

	def __init__(self, database):
		# A list of subsets of graph identifiers.
		# Is used to specify different groups of graphs (classes and training/test sets).
		# The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
		# in which the examined pattern is present.
		self.gid_subsets = []

		self.database = database  # A graphdatabase instance: contains the data for the problem.

	def store(self, dfs_code, gid_subsets):
		"""
		Code to be executed to store the pattern, if desired.
		The function will only be called for patterns that have not been pruned.
		In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
		:param dfs_code: the dfs code of the pattern (as a string).
		:param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
		"""
		print("Please implement the store function in a subclass for a specific mining task!")

	def prune(self, gid_subsets):
		"""
		prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
		should be pruned.
		:param gid_subsets: A list of the cover of the pattern for each subset.
		:return: true if the pattern should be pruned, false otherwise.
		"""
		print("Please implement the prune function in a subclass for a specific mining task!")


#####################################################################
#                             Point 1                             ##
#####################################################################

class FindingKBestSubgraphs(PatternGraphs):
	"""
	Find the top k most confident (on the positive class) frequent subgraphs.
	https://waytolearnx.com/2019/04/recuperer-une-cle-dans-un-dictionnaire-a-partir-dune-valeur-en-python.html
	"""

	def __init__(self, minsup, k, database, subsets):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		"""
		super().__init__(database)
		self.patterns = {}  # The patterns found in the end (as dfs codes represented by strings) with their cover (as
		# a list of graph ids).
		self.minsup = minsup
		self.k = k
		self.gid_subsets = subsets
		self.dico = {}  # Save k best positive confidences
		for i in range(k):
			self.dico[i + 1] = [0, 0]

	# Stores any pattern found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		# We calculate the positive confidence
		pos_support = len(gid_subsets[0])
		neg_support = len(gid_subsets[2])
		tot_support = pos_support + neg_support
		pos_conf = pos_support / tot_support

		min_val = 1.0
		min_sup = 1000000
		confidences = []
		for val in self.dico.values():
			confidences.append(val[0])
			if min_val > val[0]:
				min_val = val[0]

		for val in self.dico.values():
			if val[0] == min_val and min_sup > val[1]:
				min_sup = val[1]

		if len(self.patterns) < self.k:
			# If we don't have enough 'k best patterns'
			if pos_conf in confidences:
				# If pos_conf is already in dico
				key_list = [k for (k, val) in self.dico.items() if val[0] == pos_conf]
				key_to_use = 0
				for key in key_list:
					if self.dico[key][1] == tot_support:
						key_to_use = key
						break
				if key_to_use != 0:
					# If tot_support is already in dico
					self.patterns[key_to_use].append((dfs_code, gid_subsets))
				else:
					# If tot_support is not in dico, we add in a 'free' place
					key_list = [k for (k, val) in self.dico.items() if val[0] == min_val]
					self.dico[key_list[0]] = [pos_conf, tot_support]
					self.patterns[key_list[0]] = [(dfs_code, gid_subsets)]

			else:
				# If pos_conf is no in dico, we add it in a 'free' place
				key_list = [k for (k, val) in self.dico.items() if val[0] == min_val]
				self.dico[key_list[0]] = [pos_conf, tot_support]
				self.patterns[key_list[0]] = [(dfs_code, gid_subsets)]

		elif pos_conf >= min_val:
			if pos_conf in confidences:
				# If confidence already in k best ones, we verify the support
				key_list = [k for (k, val) in self.dico.items() if val[0] == pos_conf]
				key_to_use = 0
				for key in key_list:
					if self.dico[key][1] == tot_support:
						key_to_use = key
						break
				if key_to_use != 0:
					# If tot_support os already in dico
					self.patterns[key_to_use].append((dfs_code, gid_subsets))
				else:
					# If tot_support is not in dico
					if pos_conf == min_val:
						tmp_min = 0
						key_to_use = 0
						for key, val in self.dico.items():
							if val[0] == pos_conf and tot_support > val[1]:
								tmp_min = val[1]
								key_to_use = key
							elif val[0] == pos_conf and tot_support == val[1]:
								key_to_use = key

						if key_to_use != 0 and tmp_min == 0:
							# If we have confidence and tot_support already in dico
							self.patterns[key_to_use].append((dfs_code, gid_subsets))

						elif key_to_use != 0 and tmp_min != 0:
							# If we have confidence, but tot_support bigger than support min for this confidence
							self.patterns.pop(key_to_use)
							self.dico[key_to_use] = [pos_conf, tot_support]
							self.patterns[key_to_use] = [(dfs_code, gid_subsets)]
					else:
						# if pos_conf isn't the minimum confidence, we pop the minimum one
						key_to_use = 0
						for key, val in self.dico.items():
							if val[0] == min_val and val[1] == min_sup:
								key_to_use = key
						self.patterns.pop(key_to_use)
						self.dico[key_to_use] = [pos_conf, tot_support]
						self.patterns[key_to_use] = [(dfs_code, gid_subsets)]

			else:
				# If confidence not in k best, we remove the min and we add the nwe confidence
				key_to_use = 0
				for key, val in self.dico.items():
					if val[0] == min_val and val[1] == min_sup:
						key_to_use = key

				self.patterns.pop(key_to_use)
				self.dico[key_to_use] = [pos_conf, tot_support]
				self.patterns[key_to_use] = [(dfs_code, gid_subsets)]

	# Prunes any pattern that is not frequent in the positive class
	def prune(self, gid_subsets):
		pos_support = len(gid_subsets[0])
		neg_support = len(gid_subsets[2])
		tot_support = pos_support + neg_support
		return tot_support < self.minsup

	# creates a column for a feature matrix
	@staticmethod
	def create_fm_col(all_gids, subset_gids):
		subset_gids = set(subset_gids)
		bools = []
		for i, val in enumerate(all_gids):
			if val in subset_gids:
				bools.append(1)
			else:
				bools.append(0)
		return bools

	# return a feature matrix for each subset of examples, in which the columns correspond to patterns
	# and the rows to examples in the subset.
	def get_feature_matrices(self):
		confidences = list(self.patterns.keys())

		dico = {}
		for i in range(len(confidences)):
			tmp = confidences[i]
			for pattern, gid_subsets in self.patterns[tmp]:
				pos_support = len(gid_subsets[0])
				neg_support = len(gid_subsets[2])
				tot_support = pos_support + neg_support
				conf = pos_support / tot_support
				add = 0
				for key, val in dico.items():
					if conf == val[0] and tot_support == val[1]:
						# if it already in dico
						add = 1
						break
				if add == 0:
					# If no in dico, we have to add it
					dico[tmp] = [conf, tot_support]

		sorted_d = dict(sorted(dico.items(), key=operator.itemgetter(1), reverse=True))

		tmp_patterns = []
		for cle, value in sorted_d.items():
			for pattern, gid_subsets in self.patterns[cle]:
				tmp_patterns.append((pattern, gid_subsets))

		matrices = [[] for _ in self.gid_subsets]
		for pattern, gid_subsets in tmp_patterns:
			for i, gid_subset in enumerate(gid_subsets):
				matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
		return [numpy.array(matrix).transpose() for matrix in matrices]


def point1():
	args = sys.argv
	database_file_name_pos = args[1]  # First parameter: path to positive class file
	database_file_name_neg = args[2]  # Second parameter: path to negative class file
	k = int(args[3])  # Third parameter: k
	minsup = int(args[4])  # Fourth parameter: minimum support

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(
		database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(
		database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	subsets = [pos_ids, neg_ids]  # The ids for the positive labelled graphs in the database
	task = FindingKBestSubgraphs(minsup, k, graph_database, subsets)  # Creating task
	gSpan(task).run()  # Running gSpan

	# We list all the patterns with the k-best confidence
	confidences = list(task.patterns.keys())

	dico = {}
	for i in range(len(confidences)):
		tmp = confidences[i]
		for pattern, gid_subsets in task.patterns[tmp]:
			pos_support = len(gid_subsets[0])
			neg_support = len(gid_subsets[1])
			tot_support = pos_support + neg_support
			conf = pos_support / tot_support
			add = 0
			for key, val in dico.items():
				if conf == val[0] and tot_support == val[1]:
					# if it already in dico
					add = 1
					break
			if add == 0:
				# If no in dico, we have to add it
				dico[tmp] = [conf, tot_support]

	sorted_d = dict(sorted(dico.items(), key=operator.itemgetter(1), reverse=True))

	f = open('point1.txt', 'w')
	for cle, value in sorted_d.items():
		for pattern, gid_subsets in task.patterns[cle]:
			pos_support = len(gid_subsets[0])
			neg_support = len(gid_subsets[1])
			tot_support = pos_support + neg_support
			conf = pos_support / tot_support
			string = '{} {} {}'.format(pattern, conf, tot_support)
			print('{} {} {}'.format(pattern, conf, tot_support))
			f.write(string + '\n')
	f.close()


#####################################################################
#                             Point 2                             ##
#####################################################################

def train_and_evaluate(minsup, k, database, subsets):
	task = FindingKBestSubgraphs(minsup, k, database, subsets)  # Creating task

	gSpan(task).run()  # Running gSpan

	# Creating feature matrices for training and testing:
	features = task.get_feature_matrices()
	train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
	train_labels = numpy.concatenate(
		(numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
	test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
	test_labels = numpy.concatenate(
		(numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

	classifier = DecisionTreeClassifier(random_state=1)  # Creating model object
	classifier.fit(train_fm, train_labels)  # Training model

	predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

	accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy

	confidences = list(task.patterns.keys())

	dico = {}
	for i in range(len(confidences)):
		tmp = confidences[i]
		for pattern, gid_subsets in task.patterns[tmp]:
			pos_support = len(gid_subsets[0])
			neg_support = len(gid_subsets[2])
			tot_support = pos_support + neg_support
			conf = pos_support / tot_support
			add = 0
			for key, val in dico.items():
				if conf == val[0] and tot_support == val[1]:
					# if it already in dico
					add = 1
					break
			if add == 0:
				# If no in dico, we have to add it
				dico[tmp] = [conf, tot_support]

	sorted_d = dict(sorted(dico.items(), key=operator.itemgetter(1), reverse=True))

	f = open('point2.txt', 'w')
	for cle, value in sorted_d.items():
		for pattern, gid_subsets in task.patterns[cle]:
			pos_support = len(gid_subsets[0])
			neg_support = len(gid_subsets[2])
			tot_support = pos_support + neg_support
			conf = pos_support / tot_support
			string = '{} {} {}'.format(pattern, conf, tot_support)
			print('{} {} {}'.format(pattern, conf, tot_support))
			f.write(string + '\n')
	f.close()

	# printing classification results:
	print(predicted.tolist())
	print('accuracy: {}'.format(accuracy))
	print()  # Blank line to indicate end of fold.


def point2():
	args = sys.argv
	database_file_name_pos = args[1]  # First parameter : path to positive class file
	database_file_name_neg = args[2]  # Second parameter : path to negative class file
	k = int(args[3])  # Third parameter : k
	minsup = int(args[4])  # Fourth parameter : minimum support
	nfolds = int(args[5])  # Fifith parameter : number of Folds

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(
		database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(
		database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	# If less than two folds: using the same set as training and test set (note this is not an accurate way to
	# evaluate the performances!)
	if nfolds < 2:
		subsets = [
			pos_ids,  # Positive training set
			pos_ids,  # Positive test set
			neg_ids,  # Negative training set
			neg_ids  # Negative test set
		]
		# Printing fold number:
		print('fold {}'.format(1))
		train_and_evaluate(minsup, k, graph_database, subsets)

	# Otherwise: performs k-fold cross-validation:
	else:
		pos_fold_size = len(pos_ids) // nfolds
		neg_fold_size = len(neg_ids) // nfolds
		for i in range(nfolds):
			# Use fold as test set, the others as training set for each class;
			# identify all the subsets to be maintained by the graph mining algorithm.
			subsets = [
				numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),
				# Positive training set
				pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
				numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),
				# Negative training set
				neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
			]
			# Printing fold number:
			print('fold {}'.format(i + 1))
			train_and_evaluate(minsup, k, graph_database, subsets)


#####################################################################
#                              Point 3                             ##
#####################################################################
def sequentialCovering():
	args = sys.argv
	database_file_name_pos = args[1]  # First parameter: path to positive class file
	database_file_name_neg = args[2]  # Second parameter: path to negative class file
	k = int(args[3])  # Third parameter: k
	minsup = int(args[4])  # Fourth parameter: minimum support
	nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(
		database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(
		database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids
	subsets = [pos_ids, neg_ids]  # The ids for the positive labelled graphs in the database

	rules = []
	for iteration in k:
		task = FindingKBestSubgraphs(minsup, k, graph_database, subsets)  # Creating task
		gSpan(task).run()  # Running gSpan
		confidences = list(task.patterns.keys())  # We list all the patterns with the k-best confidence


# task.patterns c'est un dico avec comme keys les confidences et en values les patterns
# Nous on veut aussi avoir le support et la transaction où se trouve le pattern


#####################################################################
#                              RUN PART                            ##
#####################################################################


if __name__ == '__main__':
	# example1()
	# example2()
	# point1()
	point2()

# POINT 2 : python main28.py data/molecules-small.pos data/molecules-small.neg 5 5 4


# POINT 1 : python main7.py data/molecules-small.pos data/molecules-small.neg 5 5
#           python main7.py data/molecules-medium.pos data/molecules-medium.neg 5 5
#           python main7.py data/molecules.pos data/molecules.neg 5 5

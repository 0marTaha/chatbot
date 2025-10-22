import os
import csv
import joblib
import random
import re
import numpy as np
import Levenshtein
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Rule-based patterns for dialog act classification
# Each pattern is a regex that matches specific dialog acts
rules = {
    "hello": [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood evening\b"],
    "bye": [r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\bfarewell\b"],
    "thankyou": [r"\bthank you\b", r"\bthanks\b", r"\bcheers\b"],
    "affirm": [r"\byes\b", r"\byep\b", r"\byeah\b", r"\bright\b", r"\bcorrect\b", r"\bsure\b"],
    "deny": [r"\bdont want\b", r"\bdon't want\b", r"\bno thanks\b", r"\bnot that\b", r"\bdont like\b", r"\bdon't like\b"],
    "confirm": [r"\bis it\b", r"\bis that\b", r"\bdo you mean\b", r"\bdid you say\b"],
    "negate": [r"\bno\b", r"\bnope\b", r"\bnot really\b"],
    "request": [r"\bwhat is\b", r"\bcan i have\b", r"\bmay i have\b", r"\bgive me\b", r"\btell me\b",
                r"\baddress\b", r"\bphone\b", r"\bpost code\b", r"\bpostcode\b"],
    "reqalts": [r"\banything else\b", r"\bany other\b", r"\banother one\b", r"\bother options\b",
                r"\bsomething else\b", r"\bhow about\b", r"\bwhat about\b"],
    "reqmore": [r"\bmore\b", r"\bshow me more\b"],
    "restart": [r"\bstart over\b", r"\brestart\b"],
    "ack": [r"\bok\b", r"\bokay\b", r"\balright\b", r"\bmmhm\b"],
    "repeat": [r"\brepeat\b", r"\bsay again\b"],
    "null": [r"\bnoise\b", r"\bunintelligible\b", r"\bcough\b", r"\bsil\b"]
}

def rule_based_predict(utterance: str):
    """
    Predict dialog act using rule-based pattern matching.
    
    Args:
        utterance (str): User input to classify
        
    Returns:
        tuple: (predicted_label, matched_pattern) or ("inform", None) if no match
    """
    utterance = utterance.lower()
    for label, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, utterance):
                return label, pattern
    return "inform", None


# Dataset loading / deduplication

def load_dataset(path="dialog_acts.dat"):
    """
    Load dialog acts dataset from file.
    
    Args:
        path (str): Path to dialog acts data file
        
    Returns:
        tuple: (utterances_array, labels_array)
    """
    data = np.genfromtxt(path, dtype=str, delimiter="\n")
    labels = []
    utts = []
    for sentence in data:
        tokens = sentence.split()
        labels.append(tokens[0])
        utts.append(" ".join(tokens[1:]).lower())
    return np.array(utts), np.array(labels)

def deduplicate(utts, labels):
    """
    Remove duplicate utterances from dataset, keeping first occurrence.
    
    Args:
        utts (array): Array of utterances
        labels (array): Array of corresponding labels
        
    Returns:
        tuple: (deduplicated_utterances, deduplicated_labels)
    """
    seen = {}
    utts_d, labels_d = [], []
    for u, l in zip(utts, labels):
        if u not in seen:
            seen[u] = l
            utts_d.append(u)
            labels_d.append(l)
    return np.array(utts_d), np.array(labels_d)

def safe_train_test_split(X, y, test_size=0.15, random_state=42):
    """
    Safely split data into train/test sets, handling cases where stratification isn't possible.
    
    Args:
        X (array): Features
        y (array): Labels
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    label_counts = Counter(y)
    if min(label_counts.values()) < 2:
        stratify = None
    else:
        stratify = y

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

# Training / Evaluation

def train_and_eval(utts, labels, random_state=42, verbose=True):
    """
    Train and prepare machine learning models for evaluation.
    
    Args:
        utts (array): Array of utterances
        labels (array): Array of corresponding labels
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print training information
        
    Returns:
        dict: Dictionary containing trained models and data splits
    """
    X_train, X_test, y_train, y_test = safe_train_test_split(
        utts, labels, test_size=0.15, random_state=random_state
    )

    if verbose:
        print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
        print("Train label distribution:", Counter(y_train))
        print("Test label distribution:", Counter(y_test))

    vect = CountVectorizer()
    X_train_bow = vect.fit_transform(X_train)
    X_test_bow = vect.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    dt = DecisionTreeClassifier(random_state=random_state)

    lr.fit(X_train_bow, y_train)
    dt.fit(X_train_bow, y_train)

    return {
        "vect": vect,
        "lr": lr,
        "dt": dt,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train
    }

def evaluate_systems(objs, majority_class, name="DATASET"):
    """
    Evaluate multiple classification systems and display results.
    
    Args:
        objs (dict): Dictionary containing trained models and test data
        majority_class (str): Most frequent class for majority baseline
        name (str): Name of dataset for display purposes
    """
    vect = objs["vect"]
    lr = objs["lr"]
    dt = objs["dt"]
    X_test = objs["X_test"]
    y_test = objs["y_test"]

    # Majority baseline
    y_pred_maj = [majority_class] * len(y_test)

    # Rule-based baseline
    y_pred_rule = [rule_based_predict(u)[0] for u in X_test]

    # LR + DT
    X_test_bow = vect.transform(X_test)
    y_pred_lr = lr.predict(X_test_bow)
    y_pred_dt = dt.predict(X_test_bow)

    # Print metrics
    print(f"\n=== Evaluation for {name} ===")
    print("\n--- Majority baseline ---")
    print(classification_report(y_test, y_pred_maj, zero_division=0, digits=4))
    print("\n--- Rule-based baseline ---")
    print(classification_report(y_test, y_pred_rule, zero_division=0, digits=4))
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred_lr, zero_division=0, digits=4))
    print("\n--- Decision Tree ---")
    print(classification_report(y_test, y_pred_dt, zero_division=0, digits=4))

    # Confusion matrix (LR)
    cm = confusion_matrix(y_test, y_pred_lr, labels=np.unique(y_test))
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix (Logistic Regression, {name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_results(orig_objs, dedup_objs, majority_class):
    """
    Create comparison plots of different classification systems on original vs deduplicated data.
    
    Args:
        orig_objs (dict): Results from original dataset
        dedup_objs (dict): Results from deduplicated dataset
        majority_class (str): Most frequent class for majority baseline
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    results = []

    for name, objs in [("Original", orig_objs), ("Deduplicated", dedup_objs)]:
        X_test = objs["X_test"]
        y_test = objs["y_test"]
        vect = objs["vect"]
        lr = objs["lr"]
        dt = objs["dt"]

        X_test_bow = vect.transform(X_test)

        # Majority baseline
        y_pred_maj = [majority_class]*len(y_test)
        # Rule-based
        y_pred_rule = [rule_based_predict(u)[0] for u in X_test]
        # LR
        y_pred_lr = lr.predict(X_test_bow)
        # DT
        y_pred_dt = dt.predict(X_test_bow)

        for system, y_pred in [("Majority", y_pred_maj),
                               ("Rule-based", y_pred_rule),
                               ("Logistic Regression", y_pred_lr),
                               ("Decision Tree", y_pred_dt)]:
            results.append({
                "Dataset": name,
                "System": system,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Weighted-F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
            })

    df = pd.DataFrame(results)

    # Plot Figure Z: Accuracy and F1 scores
    metrics = ["Accuracy", "Macro-F1", "Weighted-F1"]
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="System", y=metric, hue="Dataset", ax=axes[i])
        axes[i].set_title(f"{metric} per system")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")
    plt.suptitle("Evaluation metrics across baselines and classifiers")
    plt.tight_layout()
    plt.show()

# Save models

def save_models(prefix, objs, outdir="."):
    """
    Save trained models to disk.
    
    Args:
        prefix (str): Prefix for model filenames
        objs (dict): Dictionary containing trained models
        outdir (str): Output directory for saved models
    """
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(objs["vect"], os.path.join(outdir, f"{prefix}_vect.joblib"))
    joblib.dump(objs["lr"], os.path.join(outdir, f"{prefix}_lr.joblib"))
    joblib.dump(objs["dt"], os.path.join(outdir, f"{prefix}_dt.joblib"))
    print("Saved:", prefix, "models to", outdir)


# Interactive mode

def interactive_loop(objs, majority_class):
    """
    Interactive mode for testing different classification systems.
    
    Args:
        objs (dict): Dictionary containing trained models
        majority_class (str): Most frequent class for majority baseline
    """
    print("\nInteractive mode — type 'exit' to quit.")
    print("Shows: Majority | Rule-based (keyword) | LR | DT")
    vect = objs["vect"]
    lr = objs["lr"]
    dt = objs["dt"]
    while True:
        s = input("> ").strip()
        if s.lower() == "exit":
            break
        maj = majority_class
        rb_label, rb_kw = rule_based_predict(s)
        bow = vect.transform([s.lower()])
        pred_lr = lr.predict(bow)[0]
        pred_dt = dt.predict(bow)[0]
        rb_info = rb_label + (f" (matched: '{rb_kw}')" if rb_kw else "")
        print(f"Majority: {maj} | Rule-based: {rb_info} | LR: {pred_lr} | DT: {pred_dt}")


# Main

def main(save_dir=None, interactive=False):
    """
    Main function to run complete ML evaluation pipeline.
    
    Args:
        save_dir (str, optional): Directory to save trained models
        interactive (bool): Whether to run interactive testing mode
    """
    utts, labels = load_dataset()
    print(f"Total examples: {len(utts)}, Unique utterances: {len(set(utts))}")

    # === Figure X: Distribution (Original) ===
    plt.figure(figsize=(10,6))
    orig_counts = Counter(labels)
    plt.bar(orig_counts.keys(), orig_counts.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Figure X: Class distribution (Original dataset)")
    plt.xlabel("Dialog act")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Train/eval on original
    orig_objs = train_and_eval(utts, labels)
    majority_class = Counter(orig_objs["y_train"]).most_common(1)[0][0]

    # Deduplicate
    utts_d, labels_d = deduplicate(utts, labels)

    # === Figure Y: Before vs After Deduplication ===
    plt.figure(figsize=(12,6))
    # Original
    plt.subplot(1,2,1)
    plt.bar(orig_counts.keys(), orig_counts.values())
    plt.xticks(rotation=45, ha="right")
    plt.title("Original")
    # Dedup
    dedup_counts = Counter(labels_d)
    plt.subplot(1,2,2)
    plt.bar(dedup_counts.keys(), dedup_counts.values(), color="orange")
    plt.xticks(rotation=45, ha="right")
    plt.title("Deduplicated")
    plt.suptitle("Class distribution before vs after deduplication")
    plt.tight_layout()
    plt.show()

    # Train/eval on deduplicated
    dedup_objs = train_and_eval(utts_d, labels_d)

    # Evaluate
    evaluate_systems(orig_objs, majority_class, name="Original")
    evaluate_systems(dedup_objs, majority_class, name="Deduplicated")

    # Figure Z: metrics comparison
    plot_results(orig_objs, dedup_objs, majority_class)

    # Save models
    if save_dir:
        save_models("orig", orig_objs, save_dir)
        save_models("dedup", dedup_objs, save_dir)

    # Interactive mode
    if interactive:
        interactive_loop(orig_objs, majority_class)


# Dialog System Classes and Functions

class State:
    """Dialog system states for managing conversation flow."""
    START = "start"
    ASK_PREFERENCES = "ask_preferences"
    ADDITIONAL_REQ = "additional_requirements"
    CONFIRM = "confirm"
    CONTACT_INFO = "contact_info"
    GOODBYE = "goodbye"

@dataclass
class DialogContext:
    """Context object to maintain dialog state and user preferences."""
    state: str = State.START
    preferences: Dict[str, str] = field(default_factory=dict)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    last_restaurant: Dict[str, Any] = field(default_factory=dict)
    additional_requirements: Dict[str, str] = field(default_factory=dict)
    offered_suggestion: bool = False
    state_change = False

formal_templates = {
    "welcome": "Good day, welcome to the restaurant recommendation system. How may I assist you?",
    "ask_prefs": "Please specify your preferred food type, area, and price range. Example: 'I want an Italian restaurant in the south, cheap'.",
    "ask_missing": "You have mentioned {given}. Could you also specify your preference for {missing}?",
    "recommend": "I recommend {name}, which serves {food} cuisine in the {area}, priced {price}.",
    "alt": "An alternative option is {name}, serving {food} cuisine in the {area}, priced {price}.",
    "no_match": "Apologies, no restaurant matches your current preferences.",
    "no_more": "There are no more alternatives available. You may end the session.",
    "goodbye": "Goodbye, enjoy your meal!",
    "contact_intro": "Would you like information such as phone number or address? If not, you can say goodbye to end.",
    "contact_phone": "The phone number of {name} is {phone}.",
    "contact_address": "The address of {name} is {address}.",
    "ask_additional": "Do you have any additional requirements? (e.g., romantic, touristic, assigned seats, children)",
    "reasoning": "Reasoning for {name}: {reason}",
    "not_understood": "Apologies, I did not understand that. Could you rephrase?",
    "retry": "Understood. Please re-enter your references."
}

informal_templates = {
    "welcome": "Hello, welcome to the restaurant system. What can I help you with?",
    "ask_prefs": "Which food type, area and price do you prefer? Example: 'I want an Italian restaurant in the south, cheap'.",
    "ask_missing": "Could you tell me your preference for {missing}?",
    "recommend": "I recommend {name}, which serves {food} food in the {area}, priced {price}.",
    "alt": "Here is another option: {name}, which serves {food} food in the {area}, priced {price}.",
    "no_match": "Sorry, no restaurant matches your preferences.",
    "no_more": "I have no more alternatives. You can say goodbye to end.",
    "goodbye": "Goodbye, enjoy your meal!",
    "contact_intro": "Would you like information such as phone number or address? If not, you can say goodbye to end.",
    "contact_phone": "The phone number of {name} is {phone}.",
    "contact_address": "The address of {name} is {address}.",
    "ask_additional": "Do you have additional requirements? (e.g., romantic, touristic, assigned seats, children)",
    "reasoning": "Reasoning for {name}: {reason}",
    "not_understood": "Sorry, I didn’t understand that. Could you rephrase?",
    "retry": "Alright, please re-enter your preference."
}

def fuzzy_match(user_word, vocab, max_threshold=0.66):
    """
    Find the best fuzzy match for a user word in a vocabulary using Levenshtein distance.
    
    Args:
        user_word (str): Word to match
        vocab (list): Vocabulary to search in
        max_threshold (float): Minimum similarity score required for a match
        
    Returns:
        str or None: Best matching word from vocab, or None if no good match
    """
    best_match = []
    best_score = 0
    
    for kw in vocab:
        dist = Levenshtein.distance(user_word.lower(), kw.lower())
        score = 1 - dist / max(len(user_word), len(kw))
        if score > best_score:
            best_score = score
            best_match = [kw]
        elif abs(score - best_score) <= 1e-2:
            best_match.append(kw)
  
    if best_score < max_threshold:
        return None
    return random.choice(best_match)

def load_restaurants(path: str) -> List[Dict[str, str]]:
    """
    Load restaurant data from CSV file and add random properties for rule-based reasoning.
    
    Args:
        path (str): Path to restaurant CSV file
        
    Returns:
        List[Dict[str, str]]: List of restaurant dictionaries with properties
    """
    restaurants = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {
                "name": row.get("restaurantname", "").strip(),
                "food": row.get("food", "").strip().lower(),
                "area": row.get("area", "").strip().lower(),
                "price": row.get("pricerange", "").strip().lower(),
                "address": f"{row.get('address','').strip()}, {row.get('postcode','').strip()}",
                "phone": row.get("phone","").strip(),
                "food_quality": random.choice(["good", "bad"]),
                "crowdedness": random.choice(["busy", "not busy"]),
                "length_of_stay": random.choice(["long", "short"]),
                "touristic": None,
                "assigned_seats": None,
                "romantic": None,
                "children": None
            }
            restaurants.append(r)
    return restaurants

# Vocabulary for detecting when user has no preference
no_pref_vocab = ["any", "whatever", "no preference", "don't care", "none"]
def extract_preferences(utt: str, food_vocab=None, area_vocab=None, price_vocab=None):
    """
    Extract restaurant preferences from user utterance using fuzzy matching.
    
    Args:
        utt (str): User utterance
        food_vocab (list, optional): Vocabulary of food types
        area_vocab (list, optional): Vocabulary of areas
        price_vocab (list, optional): Vocabulary of price ranges
        
    Returns:
        dict: Dictionary of extracted preferences
    """
    utt = utt.lower()
    prefs = {}
    if price_vocab:
        for w in utt.split():
            match = fuzzy_match(w, price_vocab)
            if any(w in utt for w in no_pref_vocab):
                match = "none"
            if match:
                prefs["price"] = match
    if area_vocab:
        for w in utt.split():
            match = fuzzy_match(w, area_vocab)
            if any(w in utt for w in no_pref_vocab):
                match = "none"
            if match:
                prefs["area"] = match
    if food_vocab:
        for w in utt.split():
            match = fuzzy_match(w, food_vocab)
            if any(w in utt for w in no_pref_vocab):
                match = "none"
            if match:
                prefs["food"] = match
    return prefs

def extract_additional_req(utt: str):
    """
    Extract additional requirements from user utterance.
    
    Args:
        utt (str): User utterance
        
    Returns:
        dict: Dictionary of additional requirements
    """
    req_vocab = ["romantic", "touristic", "assigned seats", "children"]
    reqs = {}
    for word in utt.split():
        match = fuzzy_match(word, req_vocab)
        if match:
            reqs[match] = True
    return reqs

def lookup_restaurants(prefs: Dict[str, str], restaurants: List[Dict[str, str]]):
    """
    Find restaurants matching user preferences.
    
    Args:
        prefs (Dict[str, str]): User preferences
        restaurants (List[Dict[str, str]]): List of all restaurants
        
    Returns:
        tuple: (all_matches, first_match, remaining_matches)
    """
    matches = restaurants
    for slot, value in prefs.items():
        if value and value != "none":
            matches = [r for r in matches if r.get(slot, "").lower() == value]
    if not matches:
        return [], {}, []
    return matches, matches[0], matches[1:]

def train_or_load_classifier(model_path="dialog_model.pkl", retrain=False):
    """
    Train or load a dialog act classifier for the dialog system.
    
    Args:
        model_path (str): Path to save/load the model
        retrain (bool): Whether to retrain even if model exists
        
    Returns:
        Pipeline: Trained sklearn pipeline for dialog act classification
    """
    if not retrain and os.path.exists(model_path):
        return joblib.load(model_path)
    X = [
        "hello", "hi", "good morning",
        "bye", "goodbye", "see you",
        "another one", "different option", "next restaurant",
        "yes", "that is good", "sounds nice", "correct",
        "no", "not good", "don't like it", "wrong",
        "phone number", "postcode", "area", "address", "what is the phone number",
    ]
    y = [
        "greet", "greet", "greet",
        "goodbye", "goodbye", "goodbye",
        "another", "another", "another",
        "accept", "accept", "accept", "accept",
        "reject", "reject", "reject", "reject",
        "request_phone", "request_postcode", "request_area", "request_address", "request_phone",
    ]
    pipe = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    return pipe

def apply_rules(restaurant, user_requirements={}):
    """
    Apply rule-based reasoning to infer restaurant properties and generate explanations.
    
    Args:
        restaurant (dict): Restaurant data dictionary
        user_requirements (dict): User's additional requirements
        
    Returns:
        list: List of reasoning messages explaining the inferences
    """
    reason_msgs = []
    if restaurant["price"] == "cheap" and restaurant["food_quality"] == "good":
        restaurant["touristic"] = True
        reason_msgs.append("It is touristic because it is cheap and has good food.")
    if restaurant["food"] == "romanian":
        restaurant["touristic"] = False
        reason_msgs.append("It is not touristic because Romanian cuisine is unfamiliar to tourists.")
    if restaurant["crowdedness"] == "busy":
        restaurant["assigned_seats"] = True
        reason_msgs.append("Seats are assigned because the restaurant is busy.")
        restaurant["romantic"] = False
        reason_msgs.append("Not romantic because it is busy.")
    if restaurant["length_of_stay"] == "long":
        restaurant["children"] = False
        reason_msgs.append("Not advised for children because the stay is long.")
        if restaurant["romantic"] is None:
            restaurant["romantic"] = True
        elif restaurant["romantic"] is False:
            restaurant["romantic"] = "uncertain"
        reason_msgs.append("Romantic because the stay is long.")
    for req, val in user_requirements.items():
        restaurant[req] = val
    return reason_msgs

def update_prefs(ctx, new_prefs, ask_confirm=False):
    """
    Update dialog context with new preferences, optionally asking for confirmation.
    
    Args:
        ctx (DialogContext): Current dialog context
        new_prefs (dict): New preferences to add
        ask_confirm (bool): Whether to ask user for confirmation
        
    Returns:
        tuple: (updated_context, confirmation_success)
    """
    if (ask_confirm):
        pref_str = "\n".join([ f"{k}: {v}" for k, v in new_prefs.items() ])
        print(pref_str if pref_str != "" else "no preference identified")
        if input("Are these preferences correct? (y/n): ").strip().lower() != "y":
            return ctx, False
    
    ctx.preferences.update(new_prefs)
    return ctx, True

def state_transition(ctx: DialogContext, user_utt: str, clf, restaurants, food_vocab, area_vocab, price_vocab,
                     first_pref_suggestion=False, ask_confirm_each=False, templates=None, allow_restart=True):
    """
    Handle state transitions in the dialog system based on user input.
    
    Args:
        ctx (DialogContext): Current dialog context
        user_utt (str): User utterance
        clf: Trained classifier for dialog act prediction
        restaurants (list): List of restaurant data
        food_vocab (list): Vocabulary of food types
        area_vocab (list): Vocabulary of areas
        price_vocab (list): Vocabulary of price ranges
        first_pref_suggestion (bool): Whether to suggest after first preference
        ask_confirm_each (bool): Whether to confirm each preference
        templates (dict): Response templates to use
        allow_restart (bool): Whether to allow dialog restart
        
    Returns:
        tuple: (system_response, updated_context)
    """
    user_utt_lower = user_utt.lower()
    predicted_act = clf.predict([user_utt])[0]
    
    if predicted_act == "goodbye":
        ctx.state_change = False
        ctx.state = State.GOODBYE
        return templates["goodbye"], ctx
    
    if allow_restart and user_utt_lower in ["restart", "start over", "reset"]:
        ctx = DialogContext()
        return templates["welcome"], ctx
    
    if ctx.state == State.START:
        prefs = extract_preferences(user_utt, food_vocab, area_vocab, price_vocab)
        ctx, confirmed = update_prefs(ctx, prefs, ask_confirm=ask_confirm_each)

        if not confirmed:
            return templates["retry"], ctx
        
        ctx.state = State.ASK_PREFERENCES
        ctx.state_change = True
        
    if ctx.state == State.ASK_PREFERENCES:
        if not ctx.state_change:
            # only use vocabularies of missing attributes
            missing = [slot for slot in ["food", "area", "price"] if slot not in ctx.preferences or ctx.preferences[slot] is None]
            
            fv = food_vocab if "food" in missing else None
            av = area_vocab if "area" in missing and fv is None else None
            pv = price_vocab if "price" in missing and av is None else None

            new_prefs = extract_preferences(user_utt, food_vocab=fv, area_vocab=av, price_vocab=pv)
            ctx, confirmed = update_prefs(ctx, new_prefs, ask_confirm=ask_confirm_each)
            if not confirmed:
                return templates["retry"], ctx
        ctx.state_change = False
        
        matches, selected, remaining = lookup_restaurants(ctx.preferences, restaurants)
        if first_pref_suggestion and ctx.preferences.get("food") and not ctx.offered_suggestion:
            ctx.offered_suggestion = True
            ctx.last_restaurant = selected if selected else random.choice(restaurants)
            ctx.suggestions = remaining
            ctx.state = State.ADDITIONAL_REQ
            ctx.state_change = True
            return templates["ask_additional"], ctx
        missing = [slot for slot in ["food", "area", "price"] if slot not in ctx.preferences or ctx.preferences[slot] is None]
        if missing:
            given_prefs = [k for k in ctx.preferences.keys() if ctx.preferences[k] is not None]
            given_str = ", ".join(given_prefs) if given_prefs else "nothing"
            return templates["ask_missing"].format(given=given_str, missing=missing[0]), ctx
        if selected:
            ctx.last_restaurant = selected
            ctx.suggestions = remaining
            ctx.state = State.ADDITIONAL_REQ
            ctx.state_change = True
            return templates["ask_additional"], ctx
        else:
            ctx.preferences = {}
            return templates["no_match"] + " " + templates["ask_prefs"], ctx
        
    if ctx.state == State.ADDITIONAL_REQ:
        ctx.state_change = False
        if user_utt_lower in ["no", "none", "nothing"]:
            user_reqs = {}
        else:
            user_reqs = extract_additional_req(user_utt)
        ctx.additional_requirements.update(user_reqs)
        matches, selected, remaining = lookup_restaurants(ctx.preferences, restaurants)
        for r in matches:
            apply_rules(r, ctx.additional_requirements)
        if matches:
            final_choice = random.choice(matches)
            ctx.last_restaurant = final_choice
            ctx.state = State.CONFIRM
            ctx.state_change = True
            reason_text = " ".join(apply_rules(final_choice, ctx.additional_requirements))
            response = templates["recommend"].format(**final_choice)
            if reason_text:
                response += "\n" + templates["reasoning"].format(name=final_choice["name"], reason=reason_text)
            return response, ctx
        
        ctx.state = State.ASK_PREFERENCES
        ctx.state_change = True
        return templates["no_match"] + " " + templates["ask_prefs"], ctx
        
    if ctx.state == State.CONFIRM:
        ctx.state_change = False
        if user_utt_lower in ["yes", "yep", "sure"]:
            ctx.state = State.CONTACT_INFO
            ctx.state_change = True
            return templates["contact_intro"], ctx
        elif user_utt_lower in ["no", "nope", "another"]:
            if ctx.suggestions:
                alt = ctx.suggestions.pop(0)
                ctx.last_restaurant = alt
                return templates["alt"].format(**alt), ctx
            else:
                ctx.state = State.GOODBYE
                ctx.state_change = True
                return templates["no_more"], ctx
        elif predicted_act == "goodbye":
            ctx.state = State.GOODBYE
            ctx.state_change = True
            return templates["goodbye"], ctx
        else:
            return "Please say yes if you like it, or no/another if you don’t.", ctx
        
    if ctx.state == State.CONTACT_INFO:
        ctx.state_change = False
        rest = ctx.last_restaurant
        if "phone" in user_utt_lower:
            return f"The phone number of {rest['name']} is {rest.get('phone','N/A')}.", ctx
        elif "address" in user_utt_lower or "postcode" in user_utt_lower:
            return f"The address of {rest['name']} is {rest.get('address','N/A')} (including postcode).", ctx
        elif predicted_act == "goodbye" or user_utt_lower in ["bye", "goodbye"]:
            ctx.state = State.GOODBYE
            ctx.state_change = True
            return templates["goodbye"], ctx
        else:
            return templates["contact_intro"], ctx
        
    if ctx.state == State.GOODBYE:
        ctx.state_change = False
        return templates["goodbye"], ctx
    
    return templates["not_understood"], ctx

ACK_LIST = [
    "OK",
    "Alright",
    "Gotcha",
    "Sure",
    "Got it",
    "Okay",
    "Cool",
    "Right",
    "Understood",
    "Noted",
    "Alright then"
]

FORMAL_ACK_LIST = [
    "Understood",
    "Acknowledged",
    "Noted",
    "Confirmed",
    "Very well",
    "Certainly",
    "Affirmative",
    "I understand",
    "I see",
    "That's clear",
    "Received",
    "Duly noted",
    "I will take that into account"
]

ack_history = []
def print_response(sys_res, allow_ack=False, use_formal=False, prob=0.7):
    response = sys_res
    if allow_ack and random.random() < prob:
        full_ack_list = FORMAL_ACK_LIST if use_formal else ACK_LIST
        ack_list = [ ack for ack in full_ack_list if ack not in ack_history ]
        if len(ack_list) == 0:
            ack_history.clear()
            ack_list = full_ack_list
        
        selected_ack = ack_list[round(random.random() * (len(ack_list) - 1))]
        ack_history.append(selected_ack)
        response = f"{selected_ack}. {response}"
    
    print(response)


def get_ack_probability(default=0.7):
    ack_prob_str = input(f"With what probability should acknowledgements occur? (default {default}): ").strip()

    if not ack_prob_str:
        print(f"No input given. Using default value {default}.")
        return default

    try:
        ack_prob = float(ack_prob_str)
        if 0 <= ack_prob <= 1:
            return ack_prob
        
        print(f"Out of range (0–1). Using default value {default}.")
        return default
    except ValueError:
        print(f"Invalid input. Using default value {default}.")
        return default

def run_dialog_system(restaurants, food_vocab, area_vocab, price_vocab):
    """
    Run the interactive restaurant recommendation dialog system.
    
    Args:
        restaurants (list): List of restaurant data
        food_vocab (list): Vocabulary of food types
        area_vocab (list): Vocabulary of areas
        price_vocab (list): Vocabulary of price ranges
    """
    allow_ack = input("Do you want to enable acknowledgements? (y/n): ").strip().lower() == "y"
    ack_prob = get_ack_probability()

    first_pref_suggestion = input("Do you want to offer suggestions after first preference type? (y/n): ").strip().lower() == "y"
    ask_confirm_each = input("Do you want to ask confirmation for each preference? (y/n): ").strip().lower() == "y"
    use_formal = input("Use formal phrases? (y/n): ").strip().lower() == "y"
    allow_restart = input("Allow dialog restart? (y/n): ").strip().lower() == "y"
    retrain = input("Do you want to retrain the classifier? (y/n): ").strip().lower() == "y"
    templates_choice = formal_templates if use_formal else informal_templates
    clf = train_or_load_classifier(retrain=retrain)
    ctx = DialogContext()
    print(templates_choice["welcome"])
    while ctx.state != State.GOODBYE:
        user_utt = input("> ")
        if not user_utt:
            continue
        sys_resp, ctx = state_transition(
            ctx, user_utt, clf, restaurants, food_vocab, area_vocab, price_vocab,
            first_pref_suggestion=first_pref_suggestion,
            ask_confirm_each=ask_confirm_each,
            templates=templates_choice,
            allow_restart=allow_restart
        )
        print_response(sys_resp, allow_ack=allow_ack, use_formal=use_formal, prob=ack_prob)

if __name__ == "__main__":
    """
    Main execution block - runs both ML evaluation and dialog system.
    First performs ML analysis and model training, then starts the interactive dialog system.
    """
    # Run ML Evaluation first
    # print("=== Running ML Evaluation ===")
    # main(save_dir="./models", interactive=False)
    
    # Then run Dialog System
    print("\n=== Running Dialog System ===")
    restaurant_file = "restaurant_info.csv"
    if not os.path.exists(restaurant_file):
        print(f"Error: The file '{restaurant_file}' was not found.")
    else:
        restaurants = load_restaurants(restaurant_file)
        food_vocab = sorted(set(r["food"] for r in restaurants if r["food"]))
        area_vocab = sorted(set(r["area"] for r in restaurants if r["area"]))
        price_vocab = sorted(set(r["price"] for r in restaurants if r["price"]))
        run_dialog_system(restaurants, food_vocab, area_vocab, price_vocab)

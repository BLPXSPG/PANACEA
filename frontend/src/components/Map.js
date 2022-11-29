import React, { Component, Fragment } from "react";
import PropTypes from "prop-types";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText} from "@material-ui/core";
import {
    Switch,
    Route,
    Link,
    Redirect,
  } from "react-router-dom";

class Map extends Component {
  constructor(props) {
    super(props);
    this.state = {
      // The active selection's index
      data: [
        { name: 'London', value: 90, stanec:support,},
        { name: '鄂尔多斯', value: 50, stanec:support },
      ],
      // The suggestions that match the user's input
      filteredSuggestions: [],
      // Whether or not the suggestion list is shown
      showSuggestions: false,
      // What the user has entered
      userInput: "",
      loadstate: "",
    };
    this.handleSearchButtonPressed = this.handleSearchButtonPressed.bind(this);
  }

  handleSearchButtonPressed() {
    this.setState({ loadstate: this.loadingPage() });
    console.log("input", this.state.userInput)
    
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: this.state.userInput,
      }),
    };
    fetch("/api/input-query", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log("data get", data)
        this.props.path.push("/documents")
      });
  }


  onChange = e => {
    const { suggestions } = this.props;
    const userInput = e.currentTarget.value;

    // Filter our suggestions that don't contain the user's input
    const filteredSuggestions = suggestions.filter(
      suggestion =>
        suggestion.toLowerCase().indexOf(userInput.toLowerCase()) > -1
    );

    this.setState({
      activeSuggestion: 0,
      filteredSuggestions,
      showSuggestions: true,
      userInput: e.currentTarget.value
    });
  };

  onClick = e => {
    this.setState({
      activeSuggestion: 0,
      filteredSuggestions: [],
      showSuggestions: false,
      userInput: e.currentTarget.innerText
    });
  };

  onKeyDown = e => {
    const { activeSuggestion, filteredSuggestions } = this.state;

    // User pressed the enter key
    if (e.keyCode === 13) {
      this.setState({
        activeSuggestion: 0,
        showSuggestions: false,
        userInput: filteredSuggestions[activeSuggestion]
      });
    }
    // User pressed the up arrow
    else if (e.keyCode === 38) {
      if (activeSuggestion === 0) {
        return;
      }

      this.setState({ activeSuggestion: activeSuggestion - 1 });
    }
    // User pressed the down arrow
    else if (e.keyCode === 40) {
      if (activeSuggestion - 1 === filteredSuggestions.length) {
        return;
      }

      this.setState({ activeSuggestion: activeSuggestion + 1 });
    }
  };

  render() {
    const {
      onChange,
      onClick,
      onKeyDown,
      state: {
        activeSuggestion,
        filteredSuggestions,
        showSuggestions,
        userInput
      }
    } = this;

    let suggestionsListComponent;

    if (showSuggestions && userInput) {
      if (filteredSuggestions.length) {
        suggestionsListComponent = (
          <ul class="suggestions">
            {filteredSuggestions.map((suggestion, index) => {
              let className;

              // Flag the active suggestion with a class
              if (index === activeSuggestion) {
                className = "suggestion-active";
              }

              return (
                <li className={className} key={suggestion} onClick={onClick}>
                  {suggestion}
                </li>
              );
            })}
          </ul>
        );
      } else {
        suggestionsListComponent = (
          <div class="no-suggestions">
            <em style = {{color: "white"}}>No related suggestions, checking something new!</em>
          </div>
        );
      }
    }

    return (
      <Fragment>
        <input
          type="text"
          class="inputboxhome"
          onChange={onChange}
          onKeyDown={onKeyDown}
          value={userInput}
        />
        {suggestionsListComponent}
        <Grid item xs={12} align="center">
            <FormControl>
              <FormHelperText style = {{color: "rgba(255, 255, 255, 0.7)"}}>
                <div align="center">Enter a fact to be checked</div>
              </FormHelperText>
            </FormControl>
          </Grid>
        <div>
            <Button
              color="grey"
              variant="contained"
              style = {{marginTop:20}}
              onClick={this.handleSearchButtonPressed}
              component={Link}
            >
              GET STARTED
            </Button>
          </div>
          <Grid item xs={12} align="center">
            <p>{this.state.loadstate}</p>
          </Grid>
      </Fragment>
    );
  }
}

export default Map;







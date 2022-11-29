import React, { Component } from "react";
import DocumentList from "./DocumentList";
import SentenceList from "./SentenceList";
import DocumentDetails from "./DocumentDetails";
import SearchQueue from "./SearchQueue";
import SearchPage from "./SearchPage";
import TracePage from "./TracePage";
import PanaceaPage from "./PanaceaPage";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
  Redirect,
} from "react-router-dom";
import TweetSpread from "./TweetSpread";
import TweetGraph from "./TweetGraph";
import TweetMap from "./TweetMap";
import TweetCount from "./TweetCount";
import TweetVocab from "./TweetVocab";


export default class HomePage extends Component {
  defaultQuery = "vitamin C cures COVID-19";

  constructor(props) {
    super(props);

    this.state = {
      queryInput: this.defaultQuery,
    };

    this.handleQueryChange = this.handleQueryChange.bind(this);
    this.handleSearchButtonPressed = this.handleSearchButtonPressed.bind(this);
  }
 
  handleQueryChange(e) {
    this.setState({
      queryInput: e.target.value,
    });
  }

  handleSearchButtonPressed() {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: this.state.queryInput,
      }),
    };
    fetch("/api/input-query", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log("data get", data)
        this.props.history.push("/documents")
      });
  }



  renderHomePage() {
    return (
      <Grid container spacing={1} className="background-fog" alignItems="center" justify="center">

        <Grid item xs={12} align="center" className="search-box">

          <Grid item  align="center">
            <Typography variant="h1" compact="h1">
              
              PANACEA
            </Typography>
            <Typography variant="h5" compact="h5">
              <div align="center">PANdemic Ai Claim vEracity Assessment</div>
            </Typography>
            <Button
              color="grey"
              variant="contained"
              style = {{marginTop:20}}
              onClick={this.props.history.push("/search")}
              component={Link}
            >
              START FACT CHECKING
            </Button>
            <Button
              color="grey"
              variant="contained"
              style = {{marginTop:20}}
              onClick={this.props.history.push("/tracing")}
              component={Link}
            >
              START TRACING INFORMATION SPREAD
            </Button>
          </Grid>
        </Grid>
      </Grid>
    );
  }

  render() {
    return (
      <Router>
        <Switch>
          <Route
            exact
            path="/"
            render={() => {
              return this.renderHomePage()
            }}
          />
          <Route path="/panacea" component={PanaceaPage} />
          <Route path="/search" component={SearchPage} />
          <Route path="/queue" component={SearchQueue} />
          <Route path="/documents" component={DocumentList} />
          <Route path="/sentences" component={SentenceList} />
          <Route path="/document/:id" component={DocumentDetails} />
          <Route name="tracepage" path="/trace/:id" component={TracePage} />
          <Route path="/test" component={TweetGraph} />
        </Switch>
      </Router>
    );
  }



}


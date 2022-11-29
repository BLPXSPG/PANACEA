import React, { Component } from "react";
import DocumentList from "./DocumentList";
import SentenceList from "./SentenceList";
import DocumentDetails from "./DocumentDetails";
import SearchQueue from "./SearchQueue";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText} from "@material-ui/core";
//import { makeStyles } from '@material-ui/styles';
//import Autocomplete from '@mui/material/Autocomplete';
//import { Controller, useForm } from "react-hook-form";
//import Autocomplete from "@material-ui/lab/Autocomplete";
//import Autocomplete from "./Autocomplete";
import Navbar from "./Navbar";

import {
  Switch,
  Route,
  Link,
  Redirect,
} from "react-router-dom";
import SearchBox from "./SearchBox";



export default class SearchPage extends Component {
  defaultQuery = "vitamin C cures COVID-19";
  constructor(props) {
    super(props);

    this.state = {
      queryInput: this.defaultQuery,
      loadstate: " ",
      querychoices:  [ 'drinking lemon water prevents COVID-19',
                        'CPR Still Encouraged During COVID-19 Pandemic',
                        'masks impact the coronavirus transmission',
                        'vitamin C cures COVID-19',],
    };
    this.handleQueryChange = this.handleQueryChange.bind(this);
  }


  handleQueryChange(e) {
    const REACT_VERSION = React.version;
    console.log(e, REACT_VERSION);
    this.setState({
      //queryInput: e.target.textContent,
      queryInput: e,
    });
    console.log(this.state.queryInput)
  }

  renderSearchPage() {
    //const classes = useStyles();
    return (
      <div className="homepage">
        <Navbar></Navbar>
        <SearchBox path={this.props.history}></SearchBox>
        <div>
          <div className="footerStyle">© 2022 Copyright: Warwick NLP Group and QUML Cognitive Science Research Group</div>
        </div>
      </div>
    );
  }

  render() {
    return (
       this.renderSearchPage()
    );
  }
}


import React, { Component } from "react";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";
import { Switch, Route, Link, Redirect } from "react-router-dom";



export default class PanaceaPage extends Component {
  constructor(props) {
    super(props);
    console.log("search query", props)
    this.renderPanaceaPage = this.renderPanaceaPage.bind(this);
    this.toSearch = this.toSearch.bind(this);
    this.toTrace = this.toTrace.bind(this);
  }

  toSearch() {
    this.props.history.push("/search")
  }

  toTrace() {
    this.props.history.push("/trace")
  }

  renderPanaceaPage() {
    return (
      <Grid container spacing={1} className="background-fog" alignItems="center" justify="center">
        <Grid item xs={1} align="center">
          <div className="logo-warwick" href="www.bbc.com"></div>
          <div className="logo-qm"></div>
          <div className="logo-ukri"></div>
        </Grid>

        <Grid item xs={10} align="center" className="search-box">

          <Grid item align="center">
            <Typography variant="h1" compact="h1">
              PANACEA
            </Typography>
            <Typography variant="h5" compact="h5">
              <div align="center">PANdemic Ai Claim vEracity Assessment</div>
            </Typography>
          </Grid>

          {/* Fact Checking part */}
          <Grid item xs={8} align="center" style={{ marginTop: 50 }}>
            <Typography style={{ fontSize: 20 }}>
              <div align="center">FACT CHECKING</div>
            </Typography>
            <Typography style={{ fontSize: 12 }}>
              <div align="center">Check the realiablity of claims related to Covid</div>
            </Typography>
            <Button
              color="grey"
              variant="contained"
              style={{ marginTop: 5 }}
              onClick={this.toSearch}
              component={Link}
            >
              START CHECKING
            </Button>
          </Grid>
          {/* Infor Tracing part */}
          <Grid item xs={8} align="center" style={{ marginTop: 50 }}>
            <Typography style={{ fontSize: 20 }}>
              <div align="center">INFORMATION TRACING</div>
            </Typography>
            <Typography style={{ fontSize: 12 }}>
              <div align="center">Trace the spread of information related to Covid</div>
            </Typography>
            <Button
              color="grey"
              variant="contained"
              style={{ marginTop: 5 }}
              onClick={this.toTrace}
              component={Link}
            >
              START TRACING
            </Button>
          </Grid>

        </Grid>
        <Grid item xs={1} align="center">
        </Grid>
        <div>
          <div className="footerStyle">© 2022 Copyright: Warwick NLP Group and QUML Cognitive Science Research Group</div>
        </div>
      </Grid>
    );
  }


  render() {
    return (
      this.renderPanaceaPage()
    );
  }
}

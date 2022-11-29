import React from 'react';
import { Link } from "react-router-dom";
import { Grid, Button, Typography, FormControl, TextField, FormHelperText } from "@material-ui/core";

const Navbar = () => {
  return (

    <div className='navbar'>
      <Grid container spacing={1} alignItems="center" justify="center">
        <Grid item xs={3} align="center">
          <Link to="/search" className='navbarName' style={{}}>PANACEA</Link>
        </Grid>
      </Grid>
    </div >


  );
}
export default Navbar;
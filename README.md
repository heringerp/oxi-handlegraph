# oxiqle

oxiqle (OXIgraph-based sparQL Endpoint) is a fork of oxigraph that includes [rs-handlegraph](https://github.com/chfi/rs-handlegraph).
The purpose is to make pangenomic GFA-files accessible with SPARQL queries.

[Oxigraph](https://github.com/oxigraph/oxigraph) is a graph database implementing the [SPARQL](https://www.w3.org/TR/sparql11-overview/) standard.

When cloning this codebase, don't forget to clone the submodules using
`git clone --recursive https://github.com/oxigraph/oxigraph.git` to clone the repository including submodules or
`git submodule update --init` to add the submodules to the already cloned repository.

## Usage

Build oxi-handlegraph with:
```
cargo build --release
```

Run server:
```
./target/release/oxigraph_server serve -l <path_to_gfa>
```

Run query from CLI:
```
./target/release/oxigraph_server query -l <path_to_gfa> --results-file out.txt --query-file <path_to_sparql_query>
```

## Help

Feel free to ask [heringerp](https://github.com/heringerp) for help.
[Bug reports](https://github.com/heringerp/oxigraph/issues) are also very welcome.


## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Oxigraph by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

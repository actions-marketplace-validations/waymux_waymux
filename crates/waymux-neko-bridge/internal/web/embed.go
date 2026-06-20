// SPDX-License-Identifier: Apache-2.0

// Package web embeds the static viewer assets. The embed directive
// requires the embedded files to be inside this package's directory,
// which is why this thin re-export package exists.
package web

import (
	"embed"
	"io/fs"
)

//go:embed all:static
var staticEmbed embed.FS

// StaticFS returns the embedded viewer assets rooted at "static".
// Suitable for http.FS(StaticFS()).
func StaticFS() fs.FS {
	sub, err := fs.Sub(staticEmbed, "static")
	if err != nil {
		// Compile-time guarantee — embed directive ensures "static" exists.
		panic(err)
	}
	return sub
}

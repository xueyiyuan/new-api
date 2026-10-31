/*
Copyright (C) 2023-2026 QuantumNous

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without any warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

package controller

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/model"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// imageModelKeywords lists substrings used to recognise an image-generation
// model name when scanning a channel's model list. The first model whose name
// contains any keyword is treated as the channel's "first image model".
var imageModelKeywords = []string{
	"image", // gpt-image-2, gemini-...-image, etc.
	"dall",  // dall-e, dall·e
	"imagen",
	"flux",
	"stable-diffusion",
	"sdxl",
	"midjourney", "mj-",
	"seedream",
	"nano-banana",
	"glm-image", "cogview", "janus", "kolors", "wan",
	"ideogram", "recraft", "playground", "aura",
	"emusion", "turbos",
}

// TestChannelImage verifies a channel's image-generation capability by sending a
// real request to {channel.base_url}/v1/images/generations using the channel's
// first image model (or an explicit ?model= override) together with the channel
// key. It never hard-codes any upstream URL — the upstream is derived entirely
// from the channel configuration selected by the caller.
//
//	POST /api/channel/test_image/:id
func TestChannelImage(c *gin.Context) {
	channelID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		common.ApiError(c, err)
		return
	}

	channel, err := model.CacheGetChannel(channelID)
	if err != nil {
		channel, err = model.GetChannelById(channelID, true)
		if err != nil {
			common.ApiError(c, err)
			return
		}
	}

	override := strings.TrimSpace(c.Query("model"))
	imageModel, tried := resolveImageModel(channel, override)
	if imageModel == "" {
		c.JSON(http.StatusOK, gin.H{
			"success":      false,
			"error_code":   "NO_IMAGE_MODEL",
			"message":      "该渠道未找到可用的图像模型（image model）。请在渠道模型列表中确认包含图像模型，或用 ?model= 显式指定。",
			"tried_models": tried,
		})
		return
	}

	baseURL := strings.TrimRight(channel.GetBaseURL(), "/")
	imagePath := "/v1/images/generations"
	if strings.HasSuffix(baseURL, "/v1") {
		imagePath = "/images/generations"
	}
	if endpoint := strings.TrimSpace(c.Query("endpoint")); endpoint != "" {
		imagePath = endpoint
	}
	fullURL := baseURL + imagePath

	key, _, newAPIErr := channel.GetNextEnabledKey()
	if newAPIErr != nil {
		c.JSON(http.StatusOK, gin.H{
			"success":    false,
			"error_code": "NO_AVAILABLE_KEY",
			"message":    "该渠道没有可用的 API Key。",
			"model":      imageModel,
		})
		return
	}

	body := map[string]any{
		"model":  imageModel,
		"prompt": "A small red apple on a white background, minimalist product photo.",
		"n":      1,
		"size":   "1024x1024",
	}
	payload, _ := json.Marshal(body)

	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(payload))
	if err != nil {
		common.ApiError(c, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"success":      false,
			"error_code":   "UPSTREAM_UNREACHABLE",
			"message":      fmt.Sprintf("无法连接上游 (%s)：%s", fullURL, err.Error()),
			"model":        imageModel,
			"tried_models": tried,
		})
		return
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		c.JSON(http.StatusOK, gin.H{
			"success":      false,
			"error_code":   fmt.Sprintf("UPSTREAM_HTTP_%d", resp.StatusCode),
			"message":      fmt.Sprintf("上游返回 HTTP %d：%s", resp.StatusCode, truncate(string(respBody), 600)),
			"model":        imageModel,
			"tried_models": tried,
		})
		return
	}

	// OpenAI-compatible response: { "data": [ { "b64_json": "..." } | { "url": "..." } ] }
	b64 := gjson.GetBytes(respBody, "data.0.b64_json").String()
	url := gjson.GetBytes(respBody, "data.0.url").String()

	if b64 == "" && url == "" {
		c.JSON(http.StatusOK, gin.H{
			"success":      false,
			"error_code":   "NO_IMAGE_IN_RESPONSE",
			"message":      "上游返回 200，但响应中未包含 b64_json 或 url 字段，可能不是标准的 images 接口响应。",
			"model":        imageModel,
			"tried_models": tried,
		})
		return
	}

	if b64 == "" && url != "" {
		// Fetch the image server-side so the browser can download without CORS.
		fetched, ferr := fetchURLAsBase64(url, key, 120*time.Second)
		if ferr != nil {
			c.JSON(http.StatusOK, gin.H{
				"success":    false,
				"error_code": "IMAGE_FETCH_FAILED",
				"message":    fmt.Sprintf("上游返回的是图片 URL，但服务端拉取图片失败：%s", ferr.Error()),
				"model":      imageModel,
			})
			return
		}
		b64 = fetched
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"model":   imageModel,
		"format":  "b64_json",
		"image":   b64,
		"message": "",
	})
}

// resolveImageModel returns the image model to use for the test. When override is
// non-empty it is used directly; otherwise the channel's model list is scanned
// for the first entry that looks like an image-generation model.
func resolveImageModel(channel *model.Channel, override string) (string, []string) {
	models := channel.GetModels()
	if override != "" {
		return override, models
	}
	for _, m := range models {
		lower := strings.ToLower(m)
		for _, kw := range imageModelKeywords {
			if strings.Contains(lower, kw) {
				return m, models
			}
		}
	}
	return "", models
}

func fetchURLAsBase64(rawURL, key string, timeout time.Duration) (string, error) {
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return "", err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

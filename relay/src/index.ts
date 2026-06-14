/**
 * LAMA Push Relay — Cloudflare Worker
 *
 * Bridges desktop LAMA instances to mobile devices via Expo Push API.
 * Stores device_id → {secret_hash, tokens[]} mappings in KV.
 */

interface Env {
	LAMA_DEVICES: KVNamespace;
}

interface DeviceRecord {
	secret_hash: string;
	tokens: string[];
}

async function sha256hex(text: string): Promise<string> {
	const data = new TextEncoder().encode(text);
	const hash = await crypto.subtle.digest("SHA-256", data);
	return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsonResponse(body: object, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}

async function verifySecret(env: Env, deviceId: string, secret: string): Promise<DeviceRecord | null> {
	const raw = await env.LAMA_DEVICES.get(`device:${deviceId}`);
	if (!raw) return null;
	const record: DeviceRecord = JSON.parse(raw);
	const hash = await sha256hex(secret);
	if (hash !== record.secret_hash) return null;
	return record;
}

async function handleRegister(req: Request, env: Env): Promise<Response> {
	const { device_id, secret, push_token } = (await req.json()) as {
		device_id?: string;
		secret?: string;
		push_token?: string;
	};
	if (!device_id || !secret || !push_token) {
		return jsonResponse({ error: "Missing device_id, secret, or push_token" }, 400);
	}

	const secretHash = await sha256hex(secret);
	const key = `device:${device_id}`;
	const raw = await env.LAMA_DEVICES.get(key);

	let record: DeviceRecord;
	if (raw) {
		record = JSON.parse(raw);
		if (record.secret_hash !== secretHash) {
			return jsonResponse({ error: "Invalid secret" }, 403);
		}
		if (!record.tokens.includes(push_token)) {
			record.tokens.push(push_token);
		}
	} else {
		record = { secret_hash: secretHash, tokens: [push_token] };
	}

	await env.LAMA_DEVICES.put(key, JSON.stringify(record));
	return jsonResponse({ status: "registered", token_count: record.tokens.length });
}

async function handleUnregister(req: Request, env: Env): Promise<Response> {
	const { device_id, secret, push_token } = (await req.json()) as {
		device_id?: string;
		secret?: string;
		push_token?: string;
	};
	if (!device_id || !secret || !push_token) {
		return jsonResponse({ error: "Missing device_id, secret, or push_token" }, 400);
	}

	const record = await verifySecret(env, device_id, secret);
	if (!record) {
		return jsonResponse({ error: "Invalid device_id or secret" }, 403);
	}

	record.tokens = record.tokens.filter((t) => t !== push_token);
	if (record.tokens.length === 0) {
		await env.LAMA_DEVICES.delete(`device:${device_id}`);
	} else {
		await env.LAMA_DEVICES.put(`device:${device_id}`, JSON.stringify(record));
	}
	return jsonResponse({ status: "unregistered" });
}

async function handlePush(req: Request, env: Env): Promise<Response> {
	const { device_id, secret, title, body, data } = (await req.json()) as {
		device_id?: string;
		secret?: string;
		title?: string;
		body?: string;
		data?: Record<string, unknown>;
	};
	if (!device_id || !secret || !title || !body) {
		return jsonResponse({ error: "Missing device_id, secret, title, or body" }, 400);
	}

	const record = await verifySecret(env, device_id, secret);
	if (!record) {
		return jsonResponse({ error: "Invalid device_id or secret" }, 403);
	}
	if (record.tokens.length === 0) {
		return jsonResponse({ error: "No push tokens registered" }, 404);
	}

	const messages = record.tokens.map((token) => ({
		to: token,
		sound: "default" as const,
		title,
		body,
		data: data || {},
	}));

	const expoResp = await fetch("https://exp.host/--/api/v2/push/send", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(messages),
	});

	const expoResult = await expoResp.json();
	return jsonResponse({ status: "sent", expo: expoResult });
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (request.method === "OPTIONS") {
			return new Response(null, {
				headers: {
					"Access-Control-Allow-Origin": "*",
					"Access-Control-Allow-Methods": "POST, OPTIONS",
					"Access-Control-Allow-Headers": "Content-Type",
				},
			});
		}

		let response: Response;

		if (request.method !== "POST") {
			response = jsonResponse({ error: "Method not allowed" }, 405);
		} else if (url.pathname === "/register") {
			response = await handleRegister(request, env);
		} else if (url.pathname === "/unregister") {
			response = await handleUnregister(request, env);
		} else if (url.pathname === "/push") {
			response = await handlePush(request, env);
		} else {
			response = jsonResponse({ error: "Not found" }, 404);
		}

		// Add CORS headers to all responses
		response.headers.set("Access-Control-Allow-Origin", "*");
		return response;
	},
};

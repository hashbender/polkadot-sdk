// Copyright (C) Parity Technologies (UK) Ltd.
// This file is part of Cumulus.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

// Cumulus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Cumulus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Cumulus. If not, see <https://www.gnu.org/licenses/>.

//! Per-block hydration for the V3 unincluded segment.
//!
//! The block builder sends just headers across; the collation task calls [`hydrate_segment`] to
//! re-assemble each unincluded parablock into a [`SegmentEntry`] (body + storage proof + relay
//! data + validation-code hash) which the collator service can turn into a `SegmentCollation`.
//!
//! Both the block-builder and block-import paths write a [`StoredEntry`] keyed by parablock hash
//! at the same time they write the block, capturing the relay-parent header, session, and
//! persisted-validation-data the block was built/validated against. Hydration anchors on that
//! entry: the parablock hash on the header identifies the entry uniquely, and the relay-parent
//! identity + PVD are read straight from the store rather than re-resolved against a relay-chain
//! client whose blocks may have rotated out of view in the meantime.
//!
//! [`StoredEntry`]: cumulus_client_resubmission_store::StoredEntry

use codec::Encode;
use cumulus_client_consensus_common::ValidationCodeHashProvider;
use cumulus_client_resubmission_store::ResubmissionStore;
use cumulus_primitives_core::{relay_chain::Hash as RelayHash, PersistedValidationData};
use polkadot_primitives::ValidationCodeHash;
use sc_client_api::{backend::AuxStore, Backend};
use sp_api::StorageProof;
use sp_blockchain::{Backend as BlockchainBackend, Error as BlockchainError, HeaderBackend};
use sp_runtime::traits::{Block as BlockT, Header as HeaderT};

/// One unincluded parablock's collation-ready payload.
pub struct SegmentEntry<Block: BlockT> {
	pub relay_parent: RelayHash,
	pub parent_header: Block::Header,
	pub blocks: Vec<Block>,
	pub proof: StorageProof,
	pub validation_code_hash: ValidationCodeHash,
	pub validation_data: PersistedValidationData,
}

/// Why a single unincluded-segment header could not be hydrated into a [`SegmentEntry`].
///
/// Carries only the *cause*; the caller is expected to attach the failing block's number/hash to
/// its log line. None of these are fatal at the segment level — `hydrate_segment` skips the entry
/// and continues with the rest.
#[derive(Debug, thiserror::Error)]
pub enum HydrateError {
	/// No stored proof/relay-parent metadata in the resubmission store (pruned on finality, or
	/// never written — e.g. block was imported before the store was wired up).
	#[error("no stored storage-proof entry (entry was pruned or never written)")]
	StoredEntryMissing,
	/// The resubmission store errored on read.
	#[error("resubmission store load failed: {0}")]
	StoreLoad(BlockchainError),
	/// The parent parablock's header is not in the local backend (pruned or never imported).
	#[error("parent header not in the parachain backend")]
	ParentHeaderMissing,
	/// The parachain backend errored while looking up the parent header.
	#[error("parachain backend errored looking up parent header: {0}")]
	ParentHeaderBackend(BlockchainError),
	/// The block's body is not in the local backend.
	#[error("block body not in the parachain backend")]
	BodyMissing,
	/// The parachain backend errored while looking up the block body.
	#[error("parachain backend errored looking up block body: {0}")]
	BodyBackend(BlockchainError),
	/// No validation-code hash known at the parent parablock.
	#[error("no validation-code hash at parent")]
	NoValidationCodeHash,
}

const LOG_TARGET: &str = "consensus::slot_based::unincluded_segment";

/// Hydrate a list of unincluded-segment headers into [`SegmentEntry`]s by calling
/// [`build_entry`] on each. Headers that fail to hydrate locally are skipped — the rest of the
/// segment is preserved. The specific failure cause is reported via [`HydrateError`] and logged
/// here with the failing block's number/hash.
pub(super) fn hydrate_segment<Block, B, Client, CHP>(
	headers: Vec<Block::Header>,
	para_backend: &B,
	code_hash_provider: &CHP,
	store: &ResubmissionStore<Block, Client>,
) -> Vec<SegmentEntry<Block>>
where
	Block: BlockT,
	B: Backend<Block>,
	Client: AuxStore,
	CHP: ValidationCodeHashProvider<Block::Hash>,
{
	let mut entries = Vec::with_capacity(headers.len());
	for header in headers {
		let block_number = *header.number();
		let block_hash = header.hash();
		match build_entry(header, para_backend, code_hash_provider, store) {
			Ok(entry) => entries.push(entry),
			Err(err) => tracing::warn!(
				target: LOG_TARGET,
				?block_number,
				?block_hash,
				%err,
				"Skipping unincluded-segment entry.",
			),
		}
	}
	entries
}

/// Rebuild a [`SegmentEntry`] for one unincluded parablock.
///
/// The [`ResubmissionStore`] entry for the block's hash is the anchor: it carries the proof, the
/// relay-parent header, and the persisted-validation-data captured at build/import time. From
/// those, only the parachain-local body + parent header + validation-code hash still need to be
/// looked up here. No relay-chain client call is made — once the entry is in the store, hydration
/// is purely local.
///
/// Returns a [`HydrateError`] variant identifying *which* lookup failed; the caller (typically
/// [`hydrate_segment`]) logs it with the failing block's number/hash. All variants are recoverable
/// at the segment level: a missing entry means the historical can't be re-shipped, not that the
/// whole resubmit must abort.
pub(super) fn build_entry<Block, B, Client, CHP>(
	header: Block::Header,
	para_backend: &B,
	code_hash_provider: &CHP,
	store: &ResubmissionStore<Block, Client>,
) -> Result<SegmentEntry<Block>, HydrateError>
where
	Block: BlockT,
	B: Backend<Block>,
	Client: AuxStore,
	CHP: ValidationCodeHashProvider<Block::Hash>,
{
	let block_hash = header.hash();
	let parent_hash = *header.parent_hash();

	// Anchor: the store row keyed by the block's hash carries the proof, the relay-parent
	// header, and the PVD captured at build/import time. If it isn't there we can't resubmit.
	let stored = store
		.load(block_hash)
		.map_err(HydrateError::StoreLoad)?
		.ok_or(HydrateError::StoredEntryMissing)?;

	let relay_parent: RelayHash = stored.relay_parent_header.hash();

	let parent_header = para_backend
		.blockchain()
		.header(parent_hash)
		.map_err(HydrateError::ParentHeaderBackend)?
		.ok_or(HydrateError::ParentHeaderMissing)?;

	let body = para_backend
		.blockchain()
		.body(block_hash)
		.map_err(HydrateError::BodyBackend)?
		.ok_or(HydrateError::BodyMissing)?;
	let block = Block::new(header, body);

	let validation_code_hash =
		code_hash_provider.code_hash_at(parent_hash).ok_or(HydrateError::NoValidationCodeHash)?;

	// The stored PVD was obtained from the relay chain with `OccupiedCoreAssumption::TimedOut`
	// (see `resubmission::resolve_session_and_pvd`), so its `parent_head` is the
	// **currently-included** head at write time — correct for the first unincluded block, but
	// stale for any position-2+ block whose actual para parent is an older unincluded ancestor.
	// Validators verify against the block's true parent, so override the field with the actual
	// para parent's encoded header. The other PVD fields (`relay_parent_number`,
	// `relay_parent_storage_root`, `max_pov_size`) are properties of the relay parent and
	// remain valid regardless of which para parent we anchor on.
	let validation_data = PersistedValidationData {
		parent_head: parent_header.encode().into(),
		..stored.persisted_validation_data.clone()
	};

	Ok(SegmentEntry {
		relay_parent,
		parent_header,
		blocks: vec![block],
		proof: (*stored.proof).clone(),
		validation_code_hash,
		validation_data,
	})
}
